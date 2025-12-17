# /home/jinyang_wang/Dev/project/argue/v2/llama_cpp/llama.cpp/build/bin/llama-cli \
#   -m /home/jinyang_wang/Dev/project/argue/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
#   -p "Hello, who are you?" \
#   -n 64

/home/jinyang_wang/Dev/project/argue/v2/llama_cpp/llama.cpp/build/bin/llama-server \
  -m /home/jinyang_wang/Dev/project/argue/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  -ngl 0