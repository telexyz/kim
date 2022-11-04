https://jonathanbgn.com/2021/06/29/illustrated-wav2vec.html

https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html

- - -

https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio

https://huggingface.co/blog/fine-tune-wav2vec2-english

https://arxiv.org/pdf/2006.13979.pdf

https://distill.pub/2017/ctc (for fine-tuned)


## `wav2vec` Transformer Model

https://github.com/vietai/ASR

https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h

https://github.com/khanld/ASR-Wav2vec-Finetune


## Kim Features

* Tensor core + FP16 `matmul` to speed up on recent GPUs
* Fused Ops (manually) for both training and reference
* GroupNorm and AttentionBlock for SD

## runpod.io
https://www.runpod.io/gpu-instance/pricing

- 1x RTX 3090	 24 GB 		$0.440 / h => 11k vnd / h
- 8x RTX 3090	192 GB 		$3.520 / h
