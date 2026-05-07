"""
HMM test script — loads a saved model and predicts on test.txt.

Usage:
    python test_hmm.py              # both English and Chinese
    python test_hmm.py English
    python test_hmm.py Chinese

Prerequisites:
    Run python src/hmm_ner.py first to train and save the model weights.
    Saved model: results/hmm/hmm_model_{lang}.pkl

Input:  data/{English,Chinese}/test.txt
Output: results/hmm/hmm_test_result_{lang}.txt
"""

import os
import sys
import pickle

from hmm_ner import HMM, load_data, predict_test


def main():
    _SRC = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(_SRC)  # NER/
    languages = ['English', 'Chinese'] if len(sys.argv) < 2 else [sys.argv[1]]

    for lang in languages:
        model_path = os.path.join(data_dir, 'results', 'hmm', f'hmm_model_{lang.lower()}.pkl')
        test_path  = os.path.join(data_dir, 'data', lang, 'test.txt')
        out_path   = os.path.join(data_dir, 'results', 'hmm', f'hmm_test_result_{lang.lower()}.txt')

        if not os.path.exists(model_path):
            print(f"[{lang}] Model not found: {model_path}")
            print(f"[{lang}] Run 'python hmm_ner.py {lang}' first.")
            continue
        if not os.path.exists(test_path):
            print(f"[{lang}] Test file not found: {test_path}")
            continue

        print(f"[{lang}] Loading model from {model_path} ...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        predict_test(model, lang, test_path, out_path)
        print(f"[{lang}] Done. Predictions written to {out_path}")
        print(f"[{lang}] Evaluate all test predictions with:")
        print("  cd NER/src && python evaluate_all.py test")


if __name__ == '__main__':
    main()
