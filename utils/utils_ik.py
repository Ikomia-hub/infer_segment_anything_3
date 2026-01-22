import os
import shutil
import cv2
import torch
from huggingface_hub import hf_hub_download


def check_float16_and_bfloat16_support(cuda):
    if torch.cuda.is_available() and cuda:
        gpu = torch.device('cuda')
        compute_capability = torch.cuda.get_device_capability(gpu)
        # Compute capability 6.0 or higher
        float16_support = compute_capability[0] >= 6
        # Compute capability 8.0 or higher
        bfloat16_support = compute_capability[0] >= 8
        if bfloat16_support:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return float16_support, bfloat16_support
    else:
        print("No GPU found")
        return False, False


def resize_mask(mask, h_orig, w_orig):
    mask = cv2.resize(
        mask,
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST
    )
    return mask


def get_checkpoint_path(base_dir):
    """
    Get the checkpoint path from the weights directory.
    Downloads the checkpoint from HuggingFace if it doesn't exist locally.

    Note: SAM 3 requires authentication to download checkpoints from the Hugging Face repo.
    Before using SAM 3, please request access to the checkpoints on the SAM 3 Hugging Face repo.
    Once accepted, you need to be authenticated to download the checkpoints. You can do this by:
    1. Generating an access token at https://huggingface.co/settings/tokens
    2. Running: hf auth login (or use login() in Python)

    For more details, see: https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication

    Args:
        base_dir: Base directory of the plugin (where weights directory should be created)

    Returns:
        str: Path to the checkpoint file

    Raises:
        Exception: If download fails due to authentication or other errors
    """
    weights_dir = os.path.join(base_dir, "weights")

    # Ensure weights directory exists
    os.makedirs(weights_dir, exist_ok=True)

    checkpoint_filename = "sam3.pt"
    checkpoint_path = os.path.join(weights_dir, checkpoint_filename)

    # If checkpoint doesn't exist locally, download it
    if not os.path.isfile(checkpoint_path):
        print(f"Downloading SAM3 model weights to {checkpoint_path}...")
        try:
            SAM3_MODEL_ID = "facebook/sam3"
            SAM3_CKPT_NAME = "sam3.pt"

            # Download from HuggingFace (will use cache if available)
            # Note: This requires authentication - user must be logged in via 'hf auth login'
            # or have HF_TOKEN environment variable set
            temp_checkpoint_path = hf_hub_download(
                repo_id=SAM3_MODEL_ID,
                filename=SAM3_CKPT_NAME
            )

            # Copy to our weights directory
            shutil.copy2(temp_checkpoint_path, checkpoint_path)
            print(f"Model weights saved to {checkpoint_path}")
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "unauthorized" in error_msg or "authentication" in error_msg:
                print("\n" + "="*70)
                print("ERROR: Authentication required to download SAM 3 checkpoints.")
                print("="*70)
                print("\nBefore using SAM 3, you need to:")
                print("1. Request access to the SAM 3 checkpoints at:")
                print("   https://huggingface.co/facebook/sam3")
                print("2. Generate an access token at:")
                print("   https://huggingface.co/settings/tokens")
                print("3. Authenticate by running:")
                print("   hf auth login")
                print("   (or use login() in Python)")
                print("\nFor more details, see:")
                print(
                    "https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication")
                print("="*70 + "\n")
            else:
                print(f"\nFailed to download model weights: {e}")
                print("\nIf this is an authentication error, please ensure you:")
                print("1. Have requested access to https://huggingface.co/facebook/sam3")
                print(
                    "2. Are authenticated via 'hf auth login' or HF_TOKEN environment variable")
            raise
    else:
        print(f"Using existing model weights from {checkpoint_path}")

    return checkpoint_path
