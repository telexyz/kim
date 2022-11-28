echo !!! backend: $KIM_BACKEND, device: $KIM_DEVICE

python3 -m pytest \
	tests/test_ndarray.py \
	tests/test_nd_backend.py \
	tests/test_autograd.py \
	tests/test_init.py \
	# tests/test_conv.py \

python3 -m pytest \
	tests/test_nn.py \
	tests/test_ops.py \

KIM_BACKEND=np python3 -m pytest \
	tests/test_optim.py \

echo !!! backend: $KIM_BACKEND, device: $KIM_DEVICE
