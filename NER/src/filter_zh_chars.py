"""Stream-filter a fastText / SGNS Chinese vector file (`.gz` or plain text)
to keep only single Chinese-character lines. Useful for char-level Chinese NER:
fastText `cc.zh.300.vec` has millions of entries but only ~10k single CJK
characters, which are the only ones a char-level model can use.

Usage:
    python filter_zh_chars.py <input.vec[.gz]> <output.vec>

Output format is the same as the input (one `<token> <v1> <v2> ... <vd>` per
line); the optional fastText header is dropped, since transformer_crf_ner.py
auto-detects dimension from the first content line.
"""
import gzip
import sys


def is_single_chinese_char(token):
    if len(token) != 1:
        return False
    c = token[0]
    return '一' <= c <= '鿿'  # CJK Unified Ideographs


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    inp, out = sys.argv[1], sys.argv[2]
    opener = gzip.open if inp.endswith('.gz') else open

    n_in = n_out = 0
    dim = None
    with opener(inp, 'rt', encoding='utf-8', errors='ignore') as fin, \
         open(out, 'w', encoding='utf-8') as fout:
        first = fin.readline().rstrip().split()
        # fastText header: "<vocab> <dim>"
        if len(first) == 2 and all(p.isdigit() for p in first):
            dim = int(first[1])
        else:
            n_in += 1
            if len(first) >= 2 and is_single_chinese_char(first[0]):
                fout.write(' '.join(first) + '\n')
                n_out += 1
                dim = len(first) - 1

        for line in fin:
            n_in += 1
            parts = line.rstrip().split(' ', 1)
            if len(parts) != 2:
                continue
            if not is_single_chinese_char(parts[0]):
                continue
            fout.write(line)
            n_out += 1
            if n_in % 200000 == 0:
                print(f"  ... scanned {n_in:>8} lines, kept {n_out}",
                      file=sys.stderr)

    print(f"Done. Scanned {n_in} lines. Kept {n_out} single-char vectors. "
          f"Dim: {dim}. Output: {out}", file=sys.stderr)


if __name__ == '__main__':
    main()
