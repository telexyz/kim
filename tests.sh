echo !!! backend: $KIM_BACKEND, device: $KIM_DEVICE

python3 -m pytest \
	tests/test_optim.py \
	tests/test_ndarray.py \
	tests/test_nd_backend.py \
	tests/test_autograd.py \
	tests/test_init.py \
	tests/test_nn.py \
	tests/test_ops.py \
	tests/test_conv.py \
	tests/test_sequence_models.py \
	tests/test_simple_nn.py \
	# tests/test_mlp_resnet.py \
	# tests/test_data.py \
	# tests/test_cifar_ptb_data.py \

echo !!! backend: $KIM_BACKEND, device: $KIM_DEVICE
