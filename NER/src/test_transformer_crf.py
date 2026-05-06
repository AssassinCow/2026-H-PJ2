"""
Transformer+CRF test script — loads a saved checkpoint and predicts on test.txt.

Usage:
    python test_transformer_crf.py              # both English and Chinese
    python test_transformer_crf.py English
    python test_transformer_crf.py Chinese

Prerequisites:
    Run python transformer_crf_ner.py first to train and save the checkpoint.
    Saved checkpoint: transformer_crf_results/transformer_crf_checkpoint_{lang}.pt

Input:  English/test.txt  or  Chinese/test.txt
Output: transformer_crf_results/transformer_crf_test_result_{lang}.txt
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

from transformer_crf_ner import (
    TransformerCRF,
    NERDataset,
    make_collate_fn,
    load_data,
    decode_loader,
)


def predict_from_checkpoint(ckpt_path, test_path, output_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    language  = ckpt['language']
    vocab     = ckpt['vocab']
    tag2idx   = ckpt['tag2idx']
    idx2tag   = ckpt['idx2tag']
    char2idx  = ckpt['char2idx']
    cfg       = ckpt['model_config']

    model = TransformerCRF(
        vocab_size       = cfg['vocab_size'],
        num_tags         = cfg['num_tags'],
        d_model          = cfg['d_model'],
        nhead            = cfg['nhead'],
        num_layers       = cfg['num_layers'],
        dim_feedforward  = cfg['dim_feedforward'],
        dropout          = cfg['dropout'],
        use_casing       = cfg['use_casing'],
        use_char_cnn     = cfg['use_char_cnn'],
        num_chars        = cfg['num_chars'],
        d_char_out       = cfg['d_char_out'],
        tag2idx          = tag2idx,
        constraint_language = language,
        embedding_dropout = cfg['embedding_dropout'],
        use_pretrained   = cfg['use_pretrained'],
        pretrained_dim   = cfg['pretrained_dim'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[{language}] Checkpoint loaded from {ckpt_path}")

    test_sents = load_data(test_path)
    use_casing = cfg['use_casing']
    use_char   = cfg['use_char_cnn']
    test_ds = NERDataset(test_sents, vocab, tag2idx, char2idx, use_casing)
    collate = make_collate_fn(use_casing=use_casing, use_char=use_char)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)

    all_preds = decode_loader(model, test_loader, device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent_idx, sent in enumerate(test_sents):
            preds = all_preds[sent_idx]
            for i, (token, _) in enumerate(sent):
                f.write(f"{token} {idx2tag[preds[i]]}\n")
            f.write("\n")
    print(f"[{language}] Test predictions written to {output_path}")


def main():
    _SRC = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(_SRC)  # NER/
    languages = ['English', 'Chinese'] if len(sys.argv) < 2 else [sys.argv[1]]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    for lang in languages:
        ckpt_path = os.path.join(data_dir, 'results', 'transformer_crf',
                                 f'transformer_crf_checkpoint_{lang.lower()}.pt')
        test_path = os.path.join(data_dir, 'data', lang, 'test.txt')
        out_path  = os.path.join(data_dir, 'results', 'transformer_crf',
                                 f'transformer_crf_test_result_{lang.lower()}.txt')

        if not os.path.exists(ckpt_path):
            print(f"[{lang}] Checkpoint not found: {ckpt_path}")
            print(f"[{lang}] Run 'python transformer_crf_ner.py {lang}' first.")
            continue
        if not os.path.exists(test_path):
            print(f"[{lang}] Test file not found: {test_path}")
            continue

        predict_from_checkpoint(ckpt_path, test_path, out_path, device)
        print(f"[{lang}] Done.")
        print(f"[{lang}] Evaluate with:")
        print(f"  python -c \"from check import check; "
              f"check('{lang}', 'data/{lang}/test.txt', '{out_path}')\"")


if __name__ == '__main__':
    main()
