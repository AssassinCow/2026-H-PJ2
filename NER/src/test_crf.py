"""
CRF test script — loads a saved model and predicts on test.txt.

Usage:
    python test_crf.py              # both English and Chinese
    python test_crf.py English
    python test_crf.py Chinese

Prerequisites:
    Run python src/crf_ner.py first to train and save the model weights.
    Saved model: results/crf/crf_model_{lang}.pkl

Input:  data/{English,Chinese}/test.txt
Output: results/crf/crf_test_result_{lang}.txt
"""

import os
import sys
import pickle

from crf_ner import load_data, sent2features, sent2tokens, repair_sequence
from io_utils import write_predictions_like_input


def predict_test(crf, language, test_path, output_path):
    test_sents = load_data(test_path)
    X_test = [sent2features(s, language) for s in test_sents]
    y_pred = crf.predict(X_test)
    y_pred = [repair_sequence(seq, language) for seq in y_pred]

    write_predictions_like_input(test_path, output_path, y_pred)
    print(f"[{language}] Test predictions written to {output_path}")


def main():
    _SRC = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(_SRC)  # NER/
    languages = ['English', 'Chinese'] if len(sys.argv) < 2 else [sys.argv[1]]

    for lang in languages:
        model_path = os.path.join(data_dir, 'results', 'crf', f'crf_model_{lang.lower()}.pkl')
        test_path  = os.path.join(data_dir, 'data', lang, 'test.txt')
        out_path   = os.path.join(data_dir, 'results', 'crf', f'crf_test_result_{lang.lower()}.txt')

        if not os.path.exists(model_path):
            print(f"[{lang}] Model not found: {model_path}")
            print(f"[{lang}] Run 'python crf_ner.py {lang}' first.")
            continue
        if not os.path.exists(test_path):
            print(f"[{lang}] Test file not found: {test_path}")
            continue

        print(f"[{lang}] Loading model from {model_path} ...")
        with open(model_path, 'rb') as f:
            crf = pickle.load(f)

        predict_test(crf, lang, test_path, out_path)
        print(f"[{lang}] Done. Predictions written to {out_path}")
        print(f"[{lang}] Evaluate all test predictions with:")
        print("  python src/evaluate_all.py test")


if __name__ == '__main__':
    main()
