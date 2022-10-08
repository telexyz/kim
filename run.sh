# pip3 install pytest numpy numdifftools pybind11 requests

python3 -m pytest
# python3 -m pytest -l -v -k "forward"
# python3 -m pytest -l -v -k "backward"
# python3 -m pytest -v -k "topo_sort"
# python3 -m pytest -v -k "compute_gradient"
# python3 -m pytest -v -k "nn_softmax_loss"
# python3 -m pytest -v -k "nn_epoch"

python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -l tests/hw1_submit.py
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "forward"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "backward"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "topo_sort"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "compute_gradient"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "softmax_loss_ndl"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr -k "nn_epoch_ndl"
