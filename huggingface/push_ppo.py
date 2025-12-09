# huggingface/push_ppo.py

from huggingface_hub import HfApi, create_repo, upload_folder

def push_to_hf(repo_name="ppo-lunarlander", folder="saved_models/ppo"):
    create_repo(repo_name, exist_ok=True)
    upload_folder(
        folder_path=folder,
        repo_id=repo_name,
        commit_message="Upload PPO model"
    )
    print(f"Uploaded PPO model to https://huggingface.co/{repo_name}")