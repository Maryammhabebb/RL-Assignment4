# huggingface/push_td3.py

from huggingface_hub import HfApi, create_repo, upload_folder

def push_to_hf(repo_name="td3-lunarlander", folder="saved_models/td3"):
    create_repo(repo_name, exist_ok=True)
    upload_folder(
        folder_path=folder,
        repo_id=repo_name,
        commit_message="Upload TD3 model"
    )
    print(f"Uploaded TD3 model to https://huggingface.co/{repo_name}")
