"""
Task 1: HMM-based Named Entity Recognition
Hand-written HMM model — no ML frameworks.
Uses supervised learning to estimate parameters and Viterbi decoding for inference.
"""

import os
import sys
import math
from collections import defaultdict


# ============================================================
# Data loading
# ============================================================

def load_data(filepath):
    """Load NER data. Returns list of sentences, each a list of (token, tag)."""
    sentences = []
    sentence = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                parts = line.split(' ')
                if len(parts) == 2:
                    sentence.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    sentence.append((parts[0], 'O'))
        if sentence:
            sentences.append(sentence)
    return sentences


# ============================================================
# HMM Model
# ============================================================

class HMM:
    def __init__(self):
        self.states = []          # list of tags
        self.state2idx = {}
        self.vocab = set()

        # Log probabilities
        self.initial_prob = {}    # state -> log prob
        self.transition_prob = {} # (state_i, state_j) -> log prob
        self.emission_prob = {}   # (state, token) -> log prob

        # Smoothing parameter
        self.smooth = 1e-6

    def train(self, sentences):
        """Estimate HMM parameters from labeled sentences using MLE with smoothing."""
        # Count statistics
        initial_count = defaultdict(int)
        transition_count = defaultdict(lambda: defaultdict(int))
        emission_count = defaultdict(lambda: defaultdict(int))
        state_count = defaultdict(int)

        for sent in sentences:
            if not sent:
                continue
            # Initial state
            first_tag = sent[0][1]
            initial_count[first_tag] += 1

            for i, (token, tag) in enumerate(sent):
                state_count[tag] += 1
                emission_count[tag][token] += 1
                self.vocab.add(token)
                if i > 0:
                    prev_tag = sent[i - 1][1]
                    transition_count[prev_tag][tag] += 1

        # Build state list
        self.states = sorted(state_count.keys())
        self.state2idx = {s: i for i, s in enumerate(self.states)}
        num_states = len(self.states)
        num_sents = len(sentences)
        vocab_size = len(self.vocab)

        # Compute log probabilities with add-k smoothing
        # Initial probabilities
        for s in self.states:
            self.initial_prob[s] = math.log(
                (initial_count[s] + self.smooth) / (num_sents + self.smooth * num_states)
            )

        # Transition probabilities
        for si in self.states:
            total = sum(transition_count[si].values()) + self.smooth * num_states
            for sj in self.states:
                self.transition_prob[(si, sj)] = math.log(
                    (transition_count[si][sj] + self.smooth) / total
                )

        # Emission probabilities
        for s in self.states:
            total = sum(emission_count[s].values()) + self.smooth * (vocab_size + 1)
            for token in emission_count[s]:
                self.emission_prob[(s, token)] = math.log(
                    (emission_count[s][token] + self.smooth) / total
                )
            # Unknown token probability
            self.emission_prob[(s, '<UNK>')] = math.log(
                self.smooth / total
            )

        print(f"  States: {num_states}, Vocab: {vocab_size}, Sentences: {num_sents}")

    def _get_emission(self, state, token):
        """Get emission log probability, handling unknown tokens."""
        key = (state, token)
        if key in self.emission_prob:
            return self.emission_prob[key]
        return self.emission_prob[(state, '<UNK>')]

    def viterbi(self, tokens):
        """Viterbi decoding to find the most likely state sequence."""
        n = len(tokens)
        if n == 0:
            return []

        num_states = len(self.states)

        # dp[t][s] = log probability of best path ending in state s at time t
        # backptr[t][s] = best previous state
        dp = [{} for _ in range(n)]
        backptr = [{} for _ in range(n)]

        # Initialization (t=0)
        for s in self.states:
            dp[0][s] = self.initial_prob[s] + self._get_emission(s, tokens[0])
            backptr[0][s] = None

        # Recursion
        for t in range(1, n):
            token = tokens[t]
            for s in self.states:
                emit = self._get_emission(s, token)
                best_score = -float('inf')
                best_prev = self.states[0]
                for sp in self.states:
                    score = dp[t - 1][sp] + self.transition_prob[(sp, s)] + emit
                    if score > best_score:
                        best_score = score
                        best_prev = sp
                dp[t][s] = best_score
                backptr[t][s] = best_prev

        # Termination: find best final state
        best_final = max(self.states, key=lambda s: dp[n - 1][s])

        # Backtrack
        path = [None] * n
        path[n - 1] = best_final
        for t in range(n - 2, -1, -1):
            path[t] = backptr[t + 1][path[t + 1]]

        return path

    def predict(self, sentences):
        """Predict tags for a list of sentences."""
        results = []
        for sent in sentences:
            tokens = [t for t, _ in sent]
            tags = self.viterbi(tokens)
            results.append(tags)
        return results


# ============================================================
# Training and evaluation
# ============================================================

def train_and_predict(language, data_dir, output_path):
    """Train HMM and predict on validation set."""
    train_path = os.path.join(data_dir, language, 'train.txt')
    val_path = os.path.join(data_dir, language, 'validation.txt')

    print(f"[{language}] Loading data...")
    train_sents = load_data(train_path)
    val_sents = load_data(val_path)
    print(f"[{language}] Train: {len(train_sents)} sentences, Val: {len(val_sents)} sentences")

    print(f"[{language}] Training HMM...")
    model = HMM()
    model.train(train_sents)

    print(f"[{language}] Predicting (Viterbi decoding)...")
    y_pred = model.predict(val_sents)

    # Write prediction file
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_idx, sent in enumerate(val_sents):
            preds = y_pred[sent_idx]
            for (token, _), pred_tag in zip(sent, preds):
                f.write(f"{token} {pred_tag}\n")
            f.write("\n")
    print(f"[{language}] Predictions written to {output_path}")

    # Compute F1 manually (no sklearn)
    metrics = evaluate(language, val_sents, y_pred)

    return model, metrics


def evaluate(language, val_sents, y_pred):
    """Compute precision, recall, F1 per tag (excluding O)."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for sent_idx, sent in enumerate(val_sents):
        preds = y_pred[sent_idx]
        for (_, gold_tag), pred_tag in zip(sent, preds):
            if gold_tag == 'O' and pred_tag == 'O':
                continue
            if gold_tag == pred_tag:
                tp[gold_tag] += 1
            else:
                if pred_tag != 'O':
                    fp[pred_tag] += 1
                if gold_tag != 'O':
                    fn[gold_tag] += 1

    all_tags = sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())))

    total_tp = sum(tp[t] for t in all_tags)
    total_fp = sum(fp[t] for t in all_tags)
    total_fn = sum(fn[t] for t in all_tags)

    print()
    print("=" * 64)
    print(f"  [{language}] HMM — Per-tag results")
    print("=" * 64)
    print(f"  {'Tag':<12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'Support':>8s}")
    print("  " + "-" * 60)
    for tag in all_tags:
        p = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) > 0 else 0
        r = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        support = tp[tag] + fn[tag]
        print(f"  {tag:<12s}  {p:10.4f}  {r:10.4f}  {f1:10.4f}  {support:8d}")

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    print()
    print("+" + "=" * 62 + "+")
    print(f"|  >>> [{language}] HMM Final Score (micro avg)" + " " * (62 - 38 - len(language)) + "|")
    print("+" + "-" * 62 + "+")
    print(f"|  Precision: {micro_p:.4f}   Recall: {micro_r:.4f}   F1: {micro_f1:.4f}   " + " " * 6 + "|")
    print("+" + "=" * 62 + "+")
    print()
    return micro_p, micro_r, micro_f1


def predict_test(model, language, test_path, output_path):
    """Predict on test file (for interview)."""
    test_sents = load_data(test_path)
    y_pred = model.predict(test_sents)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_idx, sent in enumerate(test_sents):
            preds = y_pred[sent_idx]
            for (token, _), pred_tag in zip(sent, preds):
                f.write(f"{token} {pred_tag}\n")
            f.write("\n")
    print(f"[{language}] Test predictions written to {output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))

    languages = ['English', 'Chinese']
    if len(sys.argv) > 1:
        languages = [sys.argv[1]]

    summary = {}
    for lang in languages:
        output_file = os.path.join(data_dir, f'hmm_result_{lang.lower()}.txt')
        model, metrics = train_and_predict(lang, data_dir, output_file)
        summary[lang] = metrics

        # If test.txt exists, predict on it too
        test_path = os.path.join(data_dir, lang, 'test.txt')
        if os.path.exists(test_path):
            test_output = os.path.join(data_dir, f'hmm_test_result_{lang.lower()}.txt')
            predict_test(model, lang, test_path, test_output)

    # Final summary across all languages
    if len(summary) > 1:
        print()
        print("#" * 64)
        print("#  HMM — Overall Summary (micro avg on validation)")
        print("#" * 64)
        print(f"  {'Language':<12s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}")
        print("  " + "-" * 50)
        for lang, (p, r, f1) in summary.items():
            print(f"  {lang:<12s}  {p:10.4f}  {r:10.4f}  {f1:10.4f}")
        print("#" * 64)
