import os
import logging

def upload_to_huggingface(local_path, repo_id, token=None, commit_message="Update model", path_in_repo=None):
    """
    Uploads a file to the Hugging Face Hub with error handling.
    
    Args:
        local_path (str): Path to the local file (e.g., 'checkpoints/model.pth').
        repo_id (str): Hugging Face repo ID (e.g., 'username/repo-name').
        token (str, optional): HF API token. Defaults to os.getenv('HF_TOKEN').
        commit_message (str, optional): Message for the commit.
        path_in_repo (str, optional): Filename in the repo. Defaults to local filename.
    """
    # 1. Resolve Token
    token = token or os.getenv("HF_TOKEN")
    
    if not token:
        print(f"[HF] ⚠ Skipping upload: HF_TOKEN environment variable not set.")
        print(f"[HF] File saved locally at: {local_path}")
        return

    # 2. Resolve Remote Path
    if path_in_repo is None:
        path_in_repo = os.path.basename(local_path)

    # 3. Attempt Upload
    try:
        from huggingface_hub import upload_file
        
        print(f"[HF] ⤒ Uploading {local_path} to {repo_id}...")
        
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token
        )
        print(f"[HF] ✓ Successfully uploaded to: https://huggingface.co/{repo_id}")
        
    except ImportError:
        print("[HF] ✘ Error: 'huggingface_hub' library is not installed. Run `pip install huggingface_hub`.")
    except Exception as e:
        print(f"[HF] ✘ Upload failed: {e}")
        print(f"[HF] File remains saved locally at: {local_path}")


def upload_lunarlander_models(local_dir=None, repo_id=None, token=None, commit_message="Upload LunarLander models", path_in_repo_prefix=None):
    """
    Find and upload LunarLander model files from `local_dir` (or common locations) to a Hugging Face repo.

    Args:
        local_dir (str, optional): Directory containing model files. If None, common locations are searched.
        repo_id (str): Hugging Face repo id (e.g. 'username/repo'). Required.
        token (str, optional): HF token. Falls back to `HF_TOKEN` env var.
        commit_message (str, optional): Commit message used for each uploaded file.
        path_in_repo_prefix (str, optional): Optional prefix within the repo to store files under.
    """
    import glob
    import os

    token = token or os.getenv("HF_TOKEN")
    if repo_id is None:
        print("[HF] ✘ repo_id is required (e.g. 'username/repo-name').")
        return

    # Candidate directories to search if none provided
    candidates = [] if local_dir else [
        "saved_models/lunarlander",
        "saved_models/ppo/lunarlander",
        "saved_models/lunarlander_models",
        "saved_models",
    ]
    if local_dir:
        candidates = [local_dir]

    # File patterns to consider as model artifacts
    patterns = ["*.pth", "*.pt", "*.ckpt", "*.tar", "*.zip", "*.npz"]

    found_files = []
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            found_files.extend(glob.glob(os.path.join(d, pat)))

    # If nothing found and user provided an exact file path, check that too
    if not found_files and local_dir and os.path.isfile(local_dir):
        found_files.append(local_dir)

    if not found_files:
        print(f"[HF] ⚠ No model files found in candidates: {candidates}")
        return

    # Upload each file
    for filepath in sorted(found_files):
        filename = os.path.basename(filepath)
        if path_in_repo_prefix:
            path_in_repo = os.path.join(path_in_repo_prefix, filename).replace('\\', '/')
        else:
            path_in_repo = filename
        print(f"[HF] Preparing to upload: {filepath} -> {repo_id}/{path_in_repo}")
        upload_to_huggingface(filepath, repo_id, token=token, commit_message=commit_message, path_in_repo=path_in_repo)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload LunarLander model files to Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="Hugging Face repo id (e.g. username/repo)")
    parser.add_argument("--dir", default=None, help="Local directory containing model files (optional)")
    parser.add_argument("--token", default=None, help="Hugging Face token (optional, falls back to HF_TOKEN env)")
    parser.add_argument("--prefix", default=None, help="Path prefix inside the repo (optional)")
    parser.add_argument("--message", default="Upload LunarLander models", help="Commit message")
    args = parser.parse_args()

    upload_lunarlander_models(local_dir=args.dir, repo_id=args.repo, token=args.token, commit_message=args.message, path_in_repo_prefix=args.prefix)