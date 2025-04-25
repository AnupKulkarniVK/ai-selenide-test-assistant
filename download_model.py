from huggingface_hub import snapshot_download

# This will pull _all_ files (weights, tokenizer, config, etc.)
snapshot_download(
    repo_id="codellama/CodeLlama-7b-Instruct-hf",
    local_dir="code_models/CodeLlama-7b-Instruct-hf",
    library_name="transformers"
)