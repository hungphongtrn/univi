# Train First With A Four-Source Image Mixture

Univi will start with LoRA fine-tuning on a four-source image-input mixture: text-compressed image transcription, audio transcription from spectrogram images, image-description pairs, and Valor32k-AVQA examples. This replaces the earlier zero-shot calibration plan because the core hypothesis is about whether training can unify text, audio, and vision through visual inputs, not whether the pretrained baseline already handles those renderings without adaptation.
