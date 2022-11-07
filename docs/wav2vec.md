Forced Align

https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html

=> https://colab.research.google.com/drive/1MZuLxPq10SdqbvJRuA1jy4WJ6I3I6l33


```py
Wav2Vec2ASRBundle(
	_path='wav2vec2_fairseq_base_ls960_asr_ls960.pth', 
	_params={
		'extractor_mode': 'group_norm', 
		'extractor_conv_layer_config': [(512, 10, 5), (512, 3, 2), (512, 3, 2), 
			(512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)], 

		'extractor_conv_bias': False, 
		'encoder_embed_dim': 768,
		'encoder_projection_dropout': 0.1,
		'encoder_pos_conv_kernel': 128,
		'encoder_pos_conv_groups': 16,
		'encoder_num_layers': 12,
		'encoder_num_heads': 12,
		'encoder_attention_dropout': 0.1,
		'encoder_ff_interm_features': 3072, 
		'encoder_ff_interm_dropout': 0.0, 
		'encoder_dropout': 0.1, 
		'encoder_layer_norm_first': False, 
		'encoder_layer_drop': 0.05, 
		'aux_num_out': 29
	}, 
	_sample_rate=16000, 
	_remove_aux_axis=(1, 2, 3),
	_labels=('|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z'),
)


Wav2Vec2Model(
  (feature_extractor): FeatureExtractor(
    (conv_layers): ModuleList(
      (0): ConvLayerBlock(
        (layer_norm): GroupNorm(512, 512, eps=1e-05, affine=True)
        (conv): Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
      )
      (1): ConvLayerBlock(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
      )
      (2): ConvLayerBlock(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
      )
      (3): ConvLayerBlock(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
      )
      (4): ConvLayerBlock(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
      )
      (5): ConvLayerBlock(
        (conv): Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
      )
      (6): ConvLayerBlock(
        (conv): Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
      )
    )
  )
  (encoder): Encoder(
    (feature_projection): FeatureProjection(
      (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (projection): Linear(in_features=512, out_features=768, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (transformer): Transformer(
      (pos_conv_embed): ConvolutionalPositionalEmbedding(
        (conv): Conv1d(768, 768, kernel_size=(128,), stride=(1,), padding=(64,), groups=16)
      )
      (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (layers): ModuleList(
        (0): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (1): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (2): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (3): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (4): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (5): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (6): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (7): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (8): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (9): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (10): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (11): EncoderLayer(
          (attention): SelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (dropout): Dropout(p=0.1, inplace=False)
          (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (feed_forward): FeedForward(
            (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_dropout): Dropout(p=0.0, inplace=False)
            (output_dense): Linear(in_features=3072, out_features=768, bias=True)
            (output_dropout): Dropout(p=0.1, inplace=False)
          )
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (aux): Linear(in_features=768, out_features=29, bias=True)
)

l = model.encoder.transformer.layers[0]
p = l.attention.k_proj
p.weight

Parameter containing:
tensor([[ 0.0580, -0.0568,  0.0330,  ..., -0.0050, -0.1220, -0.0304],
        [ 0.0441, -0.0801,  0.0235,  ...,  0.0008, -0.0618,  0.0983],
        [ 0.0362, -0.0870,  0.0410,  ...,  0.1136,  0.0830,  0.0420],
        ...,
        [-0.1092, -0.0494, -0.0070,  ..., -0.0614,  0.0079, -0.0298],
        [ 0.0465, -0.0752,  0.0013,  ...,  0.0005,  0.0136,  0.0237],
        [ 0.1094,  0.0988,  0.0285,  ...,  0.0585,  0.0823, -0.0333]],
       requires_grad=True)
```

- - -


https://jonathanbgn.com/2021/06/29/illustrated-wav2vec.html

https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html

https://www.youtube.com/watch?v=fMqYul2TvBE wav2vec2 Paper Explained
