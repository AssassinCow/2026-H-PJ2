"""
Aggregate evaluation: scan all *_result_{english,chinese}.txt files, run check.py on each,
and print a single comparison table.
"""

import os
import io
import sys
from contextlib import redirect_stdout

from check import check


MODELS = [
    ('HMM',             'hmm_result_{}.txt'),
    ('CRF',             'crf_result_{}.txt'),
    ('Transformer+CRF', 'transformer_crf_result_{}.txt'),
]

LANGUAGES = [
    ('English', 'English/validation.txt'),
    ('Chinese', 'Chinese/validation.txt'),
]


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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    rows = []  # (model, language, precision, recall, f1, status)
    for model_name, file_template in MODELS:
        for lang, gold_path in LANGUAGES:
            pred_path = file_template.format(lang.lower())
            if not os.path.exists(pred_path):
                rows.append((model_name, lang, None, None, None, 'missing'))
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
    print("#  NER Project — Evaluation Summary (micro avg F1 on validation set)")
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
    for lang_name, _ in LANGUAGES:
        lang_rows = [r for r in rows if r[1] == lang_name and r[5] == 'ok']
        if not lang_rows:
            continue
        print("+" + "=" * 56 + "+")
        title = f"  >>> Best on {lang_name}"
        print(f"|{title}" + " " * max(0, 56 - len(title)) + "|")
        print("+" + "-" * 56 + "+")
        best = max(lang_rows, key=lambda r: r[4])
        print(f"|  {best[0]:<20s} F1 = {best[4]:.4f}" + " " * (56 - 2 - 20 - 5 - 6 - 4) + "|")
        print("+" + "=" * 56 + "+")
        print()


if __name__ == '__main__':
    main()
