# Univi

Univi is a research experiment on visual modality unification: render text as images, render audio as images, and use ordinary images unchanged so a single vision-language training path can consume all answer-bearing inputs visually.

The initial baseline target is Gemma 4 E2B trained with Unsloth. The first training mixture combines text-compressed image transcription, audio transcription from spectrogram images, image-description pairs, and Valor32k-AVQA examples.

See `CONTEXT.md` for the project language and `docs/RESEARCH_PLAN.md` for the current experiment plan.
