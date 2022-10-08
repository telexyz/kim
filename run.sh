# pip3 install pytest numpy numdifftools pybind11 requests

# python3 -m pytest
python3 -m pytest -l -v -k "forward"
python3 -m pytest -l -v -k "backward"
python3 -m pytest -k "topo_sort"
python3 -m pytest -k "compute_gradient"
python3 -m pytest -k "softmax_loss_kim"
python3 -m pytest -l -k "nn_epoch_kim"
