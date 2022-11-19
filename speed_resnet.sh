time KIM_BACKEND=np python3 -m pytest tests/test_mlp_resnet.py -k "test_mlp_train_speed"
echo "^^^numpy_backend^^^"
echo ""

time KIM_BACKEND=nd KIM_DEVICE=cpu python3 -m pytest tests/test_mlp_resnet.py -k "test_mlp_train_speed"
echo "^^^cpu_backend^^^"
echo ""

time KIM_BACKEND=nd KIM_DEVICE=cuda python3 -m pytest tests/test_mlp_resnet.py -k "test_mlp_train_speed"
echo "^^^CUDA_backend^^^"
echo ""
