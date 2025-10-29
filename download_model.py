import os
import argparse
from huggingface_hub import snapshot_download

def download_model(model_type, save_dir="./models", token=None):
    """
    Download a Hugging Face model snapshot to a specified directory.

    Args:
        model_type (str): The Hugging Face repo id, e.g. "Qwen/Qwen2.5-0.5B-Instruct".
        save_dir (str): Base directory where the model will be saved. Default: "./models".
        token (str or None): Hugging Face Hub token for authentication. Default: None.
    """
    local_dir = os.path.join(save_dir, model_type.replace("/", "_"))
    snapshot_dir = snapshot_download(
        repo_id=model_type,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token
    )
    print(f"Model saved to: {snapshot_dir}")
    return snapshot_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot for offline use.")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Hugging Face repo id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--save_dir", type=str, default="./models",
                        help="Directory where the model will be saved (default: ./models)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face Hub token (optional, for private models or higher rate limits)")

    args = parser.parse_args()
    download_model(args.model_type, save_dir=args.save_dir, token=args.token)