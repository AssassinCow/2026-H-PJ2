#!/usr/bin/env bash
# Download and prepare pretrained word vectors for transformer_crf_ner.py.
#
#   English: glove.6B.300d.txt   (extracted from glove.6B.zip, ~990MB)
#   Chinese: cc.zh.300.char.vec  (single-char rows filtered out of
#                                 fastText cc.zh.300.vec.gz, ~10MB)
#
# To use a smaller / different GloVe size, change GLOVE_FILE below to
# e.g. glove.6B.100d.txt, glove.6B.200d.txt, etc.
#
# Usage:
#   bash download_pretrained.sh                # downloads into ./pretrained/
#   bash download_pretrained.sh /path/to/dir   # downloads into the given dir
#
# The script is restartable: if the final output file already exists, the
# corresponding step is skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${1:-$SCRIPT_DIR/pretrained}"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "Downloading into: $DEST_DIR"
echo

# ----------------------------------------------------------------------
# English: GloVe-100
# ----------------------------------------------------------------------
GLOVE_FILE="glove.6B.300d.txt"
GLOVE_ZIP="glove.6B.zip"
GLOVE_URLS=(
  "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
  "https://nlp.stanford.edu/data/glove.6B.zip"
  "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
)

if [ -f "$GLOVE_FILE" ]; then
  echo "[EN] $GLOVE_FILE already exists, skipping."
else
  if [ ! -f "$GLOVE_ZIP" ]; then
    echo "[EN] Downloading $GLOVE_ZIP ..."
    ok=0
    for url in "${GLOVE_URLS[@]}"; do
      echo "  trying: $url"
      if curl -L --fail --connect-timeout 15 -o "$GLOVE_ZIP" "$url"; then
        ok=1; break
      fi
      rm -f "$GLOVE_ZIP"
    done
    if [ "$ok" -ne 1 ]; then
      echo "ERROR: failed to download GloVe from all mirrors." >&2
      exit 1
    fi
  fi
  echo "[EN] Extracting $GLOVE_FILE ..."
  unzip -j -o "$GLOVE_ZIP" "$GLOVE_FILE"
  rm -f "$GLOVE_ZIP"
fi

# ----------------------------------------------------------------------
# Chinese: fastText cc.zh.300, filtered to single CJK characters
# ----------------------------------------------------------------------
ZH_OUT="cc.zh.300.char.vec"
ZH_GZ="cc.zh.300.vec.gz"
ZH_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz"

if [ -f "$ZH_OUT" ]; then
  echo "[ZH] $ZH_OUT already exists, skipping."
else
  if [ ! -f "$ZH_GZ" ]; then
    echo "[ZH] Downloading $ZH_GZ (~1.4GB compressed) ..."
    curl -L --fail --connect-timeout 15 -o "$ZH_GZ" "$ZH_URL"
  fi
  echo "[ZH] Filtering single Chinese characters via filter_zh_chars.py ..."
  python "$SCRIPT_DIR/src/filter_zh_chars.py" "$ZH_GZ" "$ZH_OUT"
  rm -f "$ZH_GZ"
fi

# ----------------------------------------------------------------------
echo
echo "Done."
echo "Files:"
ls -la "$DEST_DIR"/$GLOVE_FILE "$DEST_DIR"/$ZH_OUT
echo
echo "To enable pretrained embeddings during training:"
echo
echo "  export NER_PRETRAINED_EN=\"$DEST_DIR/$GLOVE_FILE\""
echo "  export NER_PRETRAINED_ZH=\"$DEST_DIR/$ZH_OUT\""
echo "  python transformer_crf_ner.py"
