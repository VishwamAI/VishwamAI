from huggingface_hub import HfApi, create_repo, upload_file
from safetensors.torch import save_file
import torch
import os
from typing import Dict, Optional, List
from tqdm import tqdm

class HuggingFaceUploader:
    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False
    ):
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.token = token
        
        # Verify organization access
        try:
            self.api.whoami()
            orgs = self.api.list_organizations()
            if "VishwamAI" not in [org.name for org in orgs]:
                print("Warning: You don't appear to be a member of the VishwamAI organization")
        except Exception as e:
            print(f"Error verifying organization access: {e}")
        
        # Create or get repo
        try:
            create_repo(
                repo_id,
                private=private,
                token=token,
                exist_ok=True,
                repo_type="model",
                organization="VishwamAI"
            )
        except Exception as e:
            print(f"Repository already exists or error creating: {e}")
            
    def upload_checkpoint(
        self,
        checkpoint_path: str,
        commit_message: str,
        epoch: int,
        metrics: Dict
    ):
        """Upload checkpoint to HF Hub with metadata."""
        # Convert to safetensors format
        model_path = f"{checkpoint_path}.safetensors"
        if not os.path.exists(model_path):
            state_dict = torch.load(checkpoint_path)
            save_file(state_dict, model_path)
        
        # Create metadata file
        metadata = {
            "epoch": epoch,
            "metrics": metrics,
            "format": "safetensors"
        }
        
        # Upload files
        files = [
            (model_path, f"epoch_{epoch}/model.safetensors"),
            (metadata, f"epoch_{epoch}/metadata.json")
        ]
        
        for local_path, hub_path in tqdm(files, desc="Uploading to Hub"):
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hub_path,
                repo_id=self.repo_id,
                commit_message=commit_message
            )
            
    def upload_metrics(
        self,
        metrics: Dict,
        epoch: int
    ):
        """Upload training metrics."""
        self.api.upload_file(
            path_or_fileobj=str(metrics),
            path_in_repo=f"metrics/epoch_{epoch}.json",
            repo_id=self.repo_id,
            commit_message=f"Upload metrics for epoch {epoch}"
        )
