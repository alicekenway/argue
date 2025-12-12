#!/home/jinyang_wang/miniconda3/bin/python
# from huggingface_hub import snapshot_download

# # The repo you linked
# repo_id = "Qwen/Qwen3-8B-GGUF"

# local_dir = "./models/Qwen3-8B-GGUF"  # change path if you like

# snapshot_download(
#     repo_id=repo_id,
#     local_dir=local_dir,
#     local_dir_use_symlinks=False,  # easier to see actual files
# )
# print(f"Downloaded to: {local_dir}")


from huggingface_hub import hf_hub_download

repo_id = "Qwen/Qwen3-8B-GGUF"
filename = "Qwen3-8B-Q4_K_M.gguf"  # check this in the repo's Files tab

local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir="/home/jinyang_wang/Dev/project/argue/models/Qwen3-8B-GGUF",
    local_dir_use_symlinks=False,
)

print("Model file saved at:", local_path)
