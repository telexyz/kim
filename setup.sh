sudo apt install -y python3-pip build-essential cmake
pip3 install --upgrade --no-deps git+https://github.com/dlsys10714/mugrade.git
pip3 install pytest numpy numdifftools pybind11 requests
make

# install pytorch cuda 11.7
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
