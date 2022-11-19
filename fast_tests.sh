echo !!! backend: $KIM_BACKEND, device: $KIM_DEVICE

python3 -m pytest \
	tests/test_autograd.py \
	tests/test_init.py \
	tests/test_nn.py \
	tests/test_ops.py \
	tests/test_optim.py

echo !!! backend: $KIM_BACKEND, device: $KIM_DEVICE
