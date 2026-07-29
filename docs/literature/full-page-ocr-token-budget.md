# Full-page OCR needs ~1k–4k visual tokens, or learned compression

**Confidence:** Established · **Used by:** [H13](../hypothesis/done/H13-density-ladder-long-targets.md), [H17](../hypothesis/todo/H17-raise-soft-token-budget.md), [H20](../hypothesis/todo/H20-audio-phoneme-resolution.md)

## Claim

Models that actually transcribe or answer questions over full text pages either spend on the order
of 1k–4k visual tokens per page, or spend far fewer through an *explicitly learned* compression
module. No published full-page reader uses a small **fixed** budget with a naive linear patchify.

## Evidence

| system | budget | mechanism |
|---|---|---|
| Pix2Struct (Lee et al., ICML 2023, arXiv:2210.03347) | 2048 patches | variable-resolution screenshot parsing |
| LLaVA-NeXT | 576 → 2880 | AnyRes tiling |
| Qwen2-VL (arXiv:2409.12191) | dynamic | native dynamic resolution |
| InternVL-1.5 | 256/tile × up to 40 tiles | tiling |
| mPLUG-DocOwl2 (arXiv:2409.03420) | 324/page | **learned cross-attention compression** |
| DeepSeek-OCR | ~256/page | specialized compression cascade |

## Why it matters for univi

Univi currently runs a **fixed 280 soft tokens** through a **linear** patchify — below every
published full-page reader, and without the learned compression that lets DocOwl2/DeepSeek-OCR get
away with ~256–324.

Measured on our data (2026-07-28):

- fineweb median page 2,832 chars ⇒ **10.7 chars per soft token**
- each 48px patch mixes **3.8 text lines** at font 14 after the 1024²→768² resize
- the prior-proof lane that reads at +115% runs at **0.1 chars/token** — a ~100× gap

## Caveat

This literature establishes what *working* systems spend; it does not establish that budget is the
binding constraint **for us**. [H11](../hypothesis/done/H11-position-decay-is-prior-induced.md)
found reading decay at the same token positions across a 100× density range, which budget alone does
not explain. [H13](../hypothesis/done/H13-density-ladder-long-targets.md) is the experiment that
settles it.
