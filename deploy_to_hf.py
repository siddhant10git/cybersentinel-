import os
from huggingface_hub import HfApi

def deploy():
    try:
        api = HfApi()
        # Verify authentication
        user_info = api.whoami()
        username = user_info["name"]
    except Exception as e:
        print("Error: You are not logged into Hugging Face.")
        print("Please run 'python -m huggingface_hub.cli login' in your terminal and try again.")
        return

    repo_id = f"{username}/cybersentinel"
    
    print(f"Creating Space {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True
    )
    
    print("Uploading files... This may take a minute.")
    # Upload everything in the current directory except the .git folder and cache
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[".git", "__pycache__", "deploy_to_hf.py", ".pytest_cache", ".venv", "venv"]
    )
    print(f"\n✅ Deployment successful! Your environment is building at: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    deploy()
