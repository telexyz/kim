# pip3 install pytest numpy numdifftools pybind11 requests

# https://colab.research.google.com/drive/1dnXj1U2DWtnaHanFoYa7XODKUDIjcl_6
# https://colab.research.google.com/github/dlsyscourse/hw3/blob/master/hw3.ipynb


# https://colab.research.google.com/github/dlsyscourse/hw2/blob/master/hw2.ipynb
# https://www.youtube.com/watch?v=uB81vGRrH0c
# 
# python3 -m pytest tests/test_init.py
python3 -m pytest tests/test_data.py
python3 -m pytest tests/test_optim.py
python3 -m pytest tests/test_ops.py -k "op_logsumexp_backward_5"
python3 -m pytest tests/test_nn.py
python3 -m pytest tests/test_mlp_resnet.py
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr tests/hw2_submit.py


# https://colab.research.google.com/github/dlsyscourse/hw1/blob/master/hw1.ipynb
# https://www.youtube.com/watch?v=cNADlHfHQHg
# 
# python3 -m pytest
# python3 -m pytest -l -v -k "forward"
# python3 -m pytest -l -v -k "backward"
# python3 -m pytest -v -k "topo_sort"
# python3 -m pytest -v -k "compute_gradient"
# python3 -m pytest -v -k "nn_softmax_loss"
# python3 -m pytest -v -k "nn_epoch"
# python3 -m mugrade submit _r1VOvEAgPZvLXFJ18agr tests/hw1_submit.py
