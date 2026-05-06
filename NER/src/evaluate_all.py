"""
Aggregate evaluation: scan all per-model prediction files, run check.py on each,
and print a single comparison table.

Usage:
  python evaluate_all.py            # validation set (default)
  python evaluate_all.py val        # validation set
  python evaluate_all.py test       # test set
"""

import os
import io
import sys
from contextlib import redirect_stdout

from check import check


# Per-split file conventions:
#   val  : results/<model>/<model>_result_<lang>.txt           vs data/<Lang>/validation.txt
#   test : results/<model>/<model>_test_result_<lang>.txt      vs data/<Lang>/test.txt
SPLITS = {
    'val': {
        'gold_template': 'data/{Lang}/validation.txt',
        'pred_suffix': 'result',
        'label': 'validation set',
    },
    'test': {
        'gold_template': 'data/{Lang}/test.txt',
        'pred_suffix': 'test_result',
        'label': 'test set',
    },
}

# (display name, results subdir, file prefix). Final pred path is
# results/<subdir>/<prefix>_<pred_suffix>_<lang>.txt for the chosen split.
MODELS = [
    ('HMM',             'hmm',             'hmm'),
    ('CRF',             'crf',             'crf'),
    ('Transformer+CRF', 'transformer_crf', 'transformer_crf'),
    ('Ensemble',        'ensemble',        'ensemble'),
]

LANGUAGES = ['English', 'Chinese']


def parse_micro_avg(report_text):
    """Parse the 'micro avg' line from sklearn classification_report output."""
    for line in report_text.splitlines():
        s = line.strip()
        if s.startswith('micro avg'):
            parts = s.split()
            # Format: 'micro avg' + precision + recall + f1 + support
            try:
                p = float(parts[2])
                r = float(parts[3])
                f1 = float(parts[4])
                return p, r, f1
            except (IndexError, ValueError):
                return None
    return None


def parse_split(argv):
    """Pick split from argv. Accept val/validation/test (case-insensitive)."""
    if len(argv) <= 1:
        return 'val'
    arg = argv[1].strip().lower()
    if arg in ('val', 'validation', 'dev'):
        return 'val'
    if arg == 'test':
        return 'test'
    sys.stderr.write(f"unknown split '{argv[1]}'. use 'val' or 'test'.\n")
    sys.exit(2)


def main():
    _SRC = os.path.dirname(os.path.abspath(__file__))
    here = os.path.dirname(_SRC)  # NER/
    os.chdir(here)

    split = parse_split(sys.argv)
    cfg = SPLITS[split]

    rows = []  # (model, language, precision, recall, f1, status)
    for model_name, subdir, prefix in MODELS:
        for lang in LANGUAGES:
            gold_path = cfg['gold_template'].format(Lang=lang)
            pred_path = os.path.join(
                'results', subdir,
                f"{prefix}_{cfg['pred_suffix']}_{lang.lower()}.txt",
            )
            if not os.path.exists(gold_path):
                rows.append((model_name, lang, None, None, None, 'gold-missing'))
                continue
            if not os.path.exists(pred_path):
                rows.append((model_name, lang, None, None, None, 'pred-missing'))
                continue
            buf = io.StringIO()
            with redirect_stdout(buf):
                check(language=lang, gold_path=gold_path, my_path=pred_path)
            metrics = parse_micro_avg(buf.getvalue())
            if metrics is None:
                rows.append((model_name, lang, None, None, None, 'parse-failed'))
            else:
                p, r, f1 = metrics
                rows.append((model_name, lang, p, r, f1, 'ok'))

    # Pretty-print
    print()
    print("#" * 72)
    print(f"#  NER Project — Evaluation Summary (micro avg F1 on {cfg['label']})")
    print("#" * 72)
    print()
    print(f"  {'Model':<18s}  {'Language':<10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}")
    print("  " + "-" * 66)
    for model_name, lang, p, r, f1, status in rows:
        if status == 'ok':
            print(f"  {model_name:<18s}  {lang:<10s}  {p:10.4f}  {r:10.4f}  {f1:10.4f}")
        else:
            print(f"  {model_name:<18s}  {lang:<10s}  ({status})")
    print("  " + "-" * 66)
    print()

    # Highlight box for each language
    for lang_name in LANGUAGES:
        lang_rows = [r for r in rows if r[1] == lang_name and r[5] == 'ok']
        if not lang_rows:
            continue
        print("+" + "=" * 56 + "+")
        title = f"  >>> Best on {lang_name} ({cfg['label']})"
        print(f"|{title}" + " " * max(0, 56 - len(title)) + "|")
        print("+" + "-" * 56 + "+")
        best = max(lang_rows, key=lambda r: r[4])
        print(f"|  {best[0]:<20s} F1 = {best[4]:.4f}" + " " * (56 - 2 - 20 - 5 - 6 - 4) + "|")
        print("+" + "=" * 56 + "+")
        print()


if __name__ == '__main__':
    main()
