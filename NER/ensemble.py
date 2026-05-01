"""
Three-way ensemble of HMM / CRF / Transformer+CRF predictions.

Strategy:
1. Load the three per-model prediction files (`*_result_<lang>.txt`).
2. For every token, build a per-tag score equal to the weighted sum of
   the models whose prediction matches that tag. Default weights are the
   per-model validation F1 so the strongest model carries the most
   influence (override via the WEIGHTS dict at the bottom).
3. Re-decode the whole sentence with a Viterbi pass that adds a large
   penalty on BIO / BMES illegal transitions, so the final sequence is
   guaranteed to be label-format consistent.

This script does not retrain any model; it only operates on the existing
prediction files. F1 is reported against the corresponding validation
file when available.
"""

import os
import sys
from collections import defaultdict


# ------------------------------------------------------------
# BIO / BMES legal-transition helpers (inlined to avoid pulling in PyTorch
# via `transformer_crf_ner`). Keep these definitions in sync with the
# corresponding helpers in transformer_crf_ner.py.
# ------------------------------------------------------------

def _split_tag(tag):
    if tag == 'O' or '-' not in tag:
        return 'O', None
    return tag.split('-', 1)


def legal_start(tag, language):
    prefix, _ = _split_tag(tag)
    if language == 'English':
        return prefix != 'I'
    return prefix not in {'M', 'E'}


def legal_end(tag, language):
    prefix, _ = _split_tag(tag)
    if language == 'English':
        return True
    return prefix not in {'B', 'M'}


def legal_transition(prev_tag, next_tag, language):
    prev_prefix, prev_type = _split_tag(prev_tag)
    next_prefix, next_type = _split_tag(next_tag)
    if language == 'English':
        if next_prefix == 'I':
            return prev_prefix in {'B', 'I'} and prev_type == next_type
        return True
    if prev_prefix in {'B', 'M'}:
        return next_prefix in {'M', 'E'} and prev_type == next_type
    if next_prefix in {'M', 'E'}:
        return False
    return True


# ------------------------------------------------------------
# I/O
# ------------------------------------------------------------

def load_pred(path):
    """Load a prediction (or gold) file: list of sentences, each list of (token, tag)."""
    sents, sent = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sents.append(sent)
                    sent = []
            else:
                parts = line.split(' ')
                if len(parts) == 2:
                    sent.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    sent.append((parts[0], 'O'))
        if sent:
            sents.append(sent)
    return sents


# ------------------------------------------------------------
# Constrained Viterbi over weighted-vote emission scores
# ------------------------------------------------------------

def viterbi_constrained(emissions, tags, language, penalty=-1e6):
    """emissions: list of dict {tag: score}. Returns best legal tag path."""
    n = len(emissions)
    if n == 0:
        return []

    dp = [{t: float('-inf') for t in tags} for _ in range(n)]
    bp = [{t: None for t in tags} for _ in range(n)]

    for t in tags:
        start_pen = 0.0 if legal_start(t, language) else penalty
        dp[0][t] = emissions[0].get(t, 0.0) + start_pen

    for i in range(1, n):
        for cur in tags:
            emit = emissions[i].get(cur, 0.0)
            best_score = float('-inf')
            best_prev = tags[0]
            for prev in tags:
                trans_pen = 0.0 if legal_transition(prev, cur, language) else penalty
                score = dp[i - 1][prev] + trans_pen + emit
                if score > best_score:
                    best_score = score
                    best_prev = prev
            dp[i][cur] = best_score
            bp[i][cur] = best_prev

    best_last_score = float('-inf')
    best_last = tags[0]
    for t in tags:
        end_pen = 0.0 if legal_end(t, language) else penalty
        score = dp[n - 1][t] + end_pen
        if score > best_last_score:
            best_last_score = score
            best_last = t

    path = [None] * n
    path[-1] = best_last
    for i in range(n - 2, -1, -1):
        path[i] = bp[i + 1][path[i + 1]]
    return path


# ------------------------------------------------------------
# Ensemble runner + evaluation
# ------------------------------------------------------------

def evaluate_micro(language, gold_sents, pred_sents, model_label):
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for gs, ps in zip(gold_sents, pred_sents):
        for (_, gt), (_, pt) in zip(gs, ps):
            if gt == 'O' and pt == 'O':
                continue
            if gt == pt:
                tp[gt] += 1
            else:
                if pt != 'O':
                    fp[pt] += 1
                if gt != 'O':
                    fn[gt] += 1
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    print(f"  [{language}] {model_label:<20s} P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    return p, r, f1


def ensemble(language, model_files, model_labels, weights, val_path, output_path):
    all_preds = [load_pred(f) for f in model_files]
    n_sents = len(all_preds[0])
    assert all(len(p) == n_sents for p in all_preds), "Sentence count mismatch across model outputs"

    tag_set = set()
    for preds in all_preds:
        for sent in preds:
            for _, t in sent:
                tag_set.add(t)
    tag_set.add('O')
    tags = sorted(tag_set)

    out_sents = []
    for s_idx in range(n_sents):
        per_model_sent = [m[s_idx] for m in all_preds]
        n_tok = len(per_model_sent[0])
        assert all(len(s) == n_tok for s in per_model_sent), \
            f"Token count mismatch in sentence {s_idx}"

        emissions = []
        for i in range(n_tok):
            score = defaultdict(float)
            for m_idx, s in enumerate(per_model_sent):
                score[s[i][1]] += weights[m_idx]
            emissions.append(dict(score))

        tokens = [per_model_sent[0][i][0] for i in range(n_tok)]
        path = viterbi_constrained(emissions, tags, language)
        out_sents.append(list(zip(tokens, path)))

    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in out_sents:
            for tok, tag in sent:
                f.write(f"{tok} {tag}\n")
            f.write("\n")
    print(f"[{language}] Ensemble predictions written to {output_path}")

    if val_path and os.path.exists(val_path):
        gold_sents = load_pred(val_path)
        print()
        print("=" * 64)
        print(f"  [{language}] Per-model vs Ensemble (validation, micro avg, ex-O)")
        print("=" * 64)
        for label, preds in zip(model_labels, all_preds):
            evaluate_micro(language, gold_sents, preds, label)
        evaluate_micro(language, gold_sents, out_sents, 'ENSEMBLE')
        print("=" * 64)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

# Default weights ≈ each model's validation F1 (baseline numbers from
# README). Update these to reflect the latest run if needed.
WEIGHTS = {
    'English': {'HMM': 0.7432, 'CRF': 0.9048, 'Transformer+CRF': 0.8892},
    'Chinese': {'HMM': 0.8776, 'CRF': 0.9519, 'Transformer+CRF': 0.9452},
}


if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    languages = ['English', 'Chinese']
    if len(sys.argv) > 1:
        languages = [sys.argv[1]]

    for lang in languages:
        files = [
            os.path.join(data_dir, f'hmm_result_{lang.lower()}.txt'),
            os.path.join(data_dir, f'crf_result_{lang.lower()}.txt'),
            os.path.join(data_dir, f'transformer_crf_result_{lang.lower()}.txt'),
        ]
        labels = ['HMM', 'CRF', 'Transformer+CRF']
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            print(f"[{lang}] Missing prediction files; ensemble skipped: {missing}")
            continue
        w = WEIGHTS[lang]
        weights = [w[l] for l in labels]
        out = os.path.join(data_dir, f'ensemble_result_{lang.lower()}.txt')
        val = os.path.join(data_dir, lang, 'validation.txt')
        ensemble(lang, files, labels, weights, val, out)
