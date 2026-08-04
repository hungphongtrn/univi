#!/usr/bin/env zsh
# H13 density ladder — materialize 5 prior-proof OCR rungs into ONE root so a
# single training config can mix them and eval each rung separately.
#
#   rung  letters/page  font  purpose
#   d1     25           14    replicates H07 (the recipe that read at +115%)
#   d2    100           14
#   d3    400           14
#   d4   1600           14    approaches fineweb density (2832 chars/page)
#   d5    400           40    d3's density at ~1.0 line/48px patch (isolates line-demux)
#
# NOTE: the H13 doc specifies font 52 for d5, but 400 letters at font 52 overflows
# to 2 pages on a 1024px canvas (which would confound d5 with image count). Font 40
# fits one page AND hits the doc's stated intent of ~1.0 text line per 48px patch
# exactly (line_height 48px). d1-d4 sit at ~2.8 lines/patch.
set -e
OUT=data/materialized/h13-density-v0
TRAIN=20000
VAL=500

run() {  # name n_groups font
  echo "RUNG $1 (n_groups=$2 font=$3)"
  HF_HUB_OFFLINE=1 uv run python -m data.preprocessing.random_strings \
    --out "$OUT" --subset-name "$1" --n-groups "$2" --font-size "$3" \
    --train-rows $TRAIN --val-rows $VAL
}

run randstr-d1 5   14
run randstr-d2 20  14
run randstr-d3 80  14
run randstr-d4 320 14
run randstr-d5 80  40
echo LADDER_DONE
