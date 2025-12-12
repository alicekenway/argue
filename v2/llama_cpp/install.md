git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp


# 
sudo apt update
sudo apt install -y cmake build-essential
sudo apt install -y libcurl4-openssl-dev

cmake -B build
cmake --build build --config Release