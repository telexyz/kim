pip install --pre torch==2.0.0.dev20230104+cu117 torchvision torchaudio --extra-index-url  https://download.pytorch.org/whl/nightly/cu117
pip install pyarrow fastparquet fire

cd data
cat data_yfinance.7z.aa data_yfinance.7z.ab > data_yfinance.7z

sudo apt install 7zz
7zz x data_yfinance.7z