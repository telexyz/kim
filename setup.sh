sudo apt install python3-pip build-essential cmake
pip3 install --upgrade --no-deps git+https://github.com/dlsys10714/mugrade.git
pip3 install pytest numpy numdifftools pybind11 requests
make
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
