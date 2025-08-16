from huggingface_hub import snapshot_download

# Download model to local folder "./lfm2-450m"
snapshot_download(repo_id="LiquidAI/LFM2-VL-450M", local_dir="./lfm2-450m")
