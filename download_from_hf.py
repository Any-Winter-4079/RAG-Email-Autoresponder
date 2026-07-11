from huggingface_hub import snapshot_download

model_name = input("Local model name: ").strip()

snapshot_download(
    repo_id="Edue3r4t5y6/bge-m3-MUIA",
    local_dir=f"models/{model_name}",
)
