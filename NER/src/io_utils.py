"""Shared I/O helpers for NER prediction files."""

import os


def write_predictions_like_input(input_path, output_path, predictions, tag_fn=str):
    """Write predictions while preserving the input file's blank-line layout.

    `predictions` is a list of per-sentence tag sequences. The output keeps
    the same token order and blank-line positions as `input_path`, which is
    required by the project checker.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sent_idx = 0
    tok_idx = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:
        for line_no, line in enumerate(fin, 1):
            stripped = line.strip()
            if not stripped:
                fout.write('\n')
                continue

            assert sent_idx < len(predictions), (
                f"Too many tokens in {input_path}; no prediction for line {line_no}"
            )
            seq = predictions[sent_idx]
            assert tok_idx < len(seq), (
                f"Too many tokens in sentence {sent_idx} at {input_path}:{line_no}"
            )
            token = stripped.split()[0]
            fout.write(f"{token} {tag_fn(seq[tok_idx])}\n")

            tok_idx += 1
            if tok_idx == len(seq):
                sent_idx += 1
                tok_idx = 0

    assert sent_idx == len(predictions) and tok_idx == 0, (
        f"Unused predictions remain after writing {output_path}: "
        f"sentence={sent_idx}/{len(predictions)}, token={tok_idx}"
    )
