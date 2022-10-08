# pip3 install pytest numpy numdifftools pybind11 requests

# python3 -m pytest
# python3 -m pytest -l -v -k "forward"
# python3 -m pytest -l -v -k "backward"
# python3 -m pytest -v -k "topo_sort"
# python3 -m pytest -v -k "compute_gradient"
# python3 -m pytest -v -k "nn_softmax_loss"
# python3 -m pytest -v -k "nn_epoch"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -l tests/hw1_submit.py

python3 -m pytest tests/test_data.py
python3 -m pytest tests/test_nn_and_optim.py
python3 -m pytest tests/test_op_power_scalar_and_logsumexp.py
# python3 -m pytest tests/test_init.py
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -l tests/hw2_submit.py
