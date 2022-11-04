## `wav2vec` Transformer Model

- https://github.com/vietai/ASR
- https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h


## Lyric Aligment

https://challenge.zalo.ai/portal/lyric-alignment
* The minimum accepted running speed is real time. If the total inference time is longer than the total length of audios, the result will not be evaluated.

* The inference time is measured on a server with the following specifications:
	- CPU: Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz
	- RAM: 64 GB 
	- GPU: GeForce GTX 3090.


## Kim Features

* Tensor core + FP16 `matmul`
* FlashAttention with or without Triton (for Transformer) or a part of it for Kim


## runpod.io

https://www.runpod.io/gpu-instance/pricing

- 1x RTX 3090	 24 GB 		$0.440 / h => 11k vnd / h
- 8x RTX 3090	192 GB 		$3.520 / h
