# pip3 install pytest numpy numdifftools pybind11 requests

# python3 -m pytest
# python3 -m pytest -l -v -k "forward"
# python3 -m pytest -l -v -k "backward"
# python3 -m pytest -v -k "topo_sort"
# python3 -m pytest -v -k "compute_gradient"
# python3 -m pytest -v -k "nn_softmax_loss"
# python3 -m pytest -v -k "nn_epoch"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr tests/hw1_submit.py

# python3 -m pytest tests/test_init.py
python3 -m pytest tests/test_data.py
python3 -m pytest tests/test_optim.py
python3 -m pytest tests/test_ops.py -k "op_logsumexp_backward_5"
python3 -m pytest tests/test_nn.py
python3 -m pytest tests/test_mlp_resnet.py
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr tests/hw2_submit.py
