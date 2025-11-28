import torch
import numpy as np
import os
from datetime import datetime
import sys
import lpips
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from dataloader import DIV2KDataset
from models import *
from utils.utils_DIP.denoising_utils import *
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

# Import SwinIR from the file you downloaded (or paste the class definition)
# Ensure 'network_swinir.py' is in the same folder
from models.network_swinir import SwinIR

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Path to the weight file you downloaded
MODEL_PATH = '001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth' 
IMG_PATH = 'dataset/DIV2K_valid_LR_x8/0801x8.png' # Your test image
# Define transforms (convert to tensor)

use_gpu = True

if use_gpu:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.float32


tfs = transforms.Compose([
    transforms.ToTensor()
])

# Initialize Dataset
valid_dataset = DIV2KDataset(
    hr_dir="dataset/DIV2K_valid_HR",
    lr_dir="dataset/DIV2K_valid_LR_x8",  # Check your unzipped folder name
    transform=tfs
)

# Initialize DataLoader (Batch size 1 for validation/DIP)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

model = SwinIR(upscale=8,               # Task 2.1 is x8
               in_chans=3, 
               img_size=48,             # CRITICAL FIX: s48 = 48 (Not 64)
               window_size=8, 
               img_range=1., 
               depths=[6, 6, 6, 6, 6, 6], 
               embed_dim=180, 
               num_heads=[6, 6, 6, 6, 6, 6], 
               mlp_ratio=2, 
               upsampler='pixelshuffle', # CRITICAL FIX: Classical uses 'pixelshuffle'
               resi_connection='1conv')

# Load the weights
pretrained_dict = torch.load(MODEL_PATH,weights_only=False)
param_key = 'params_ema' if 'params_ema' in pretrained_dict else 'params'
model.load_state_dict(pretrained_dict[param_key] if param_key in pretrained_dict else pretrained_dict)

model.eval()
model = model.to(DEVICE)

print("Model Loaded Successfully!")

# 3. Define the Inference Function (Handles Padding)
def run_swinir(img_lr_tensor, img_hr_tensor, loss_fn, image_idx, writer):
    """
    SwinIR requires image dimensions to be multiples of the window_size (8).
    We pad the image, run inference, and then unpad it.
    """
    _, _, h, w = img_lr_tensor.size()
    
    # Calculate padding needed
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    # Pad image (Reflect padding works best to avoid border artifacts)
    img_lr_tensor = torch.nn.functional.pad(img_lr_tensor, (0, pad_w, 0, pad_h), 'reflect')

    with torch.no_grad():
        output = model(img_lr_tensor)

    # Unpad (Crop back to original size * upscale_factor)
    output = output[:, :, :h*8, :w*8]

    # --- CRITICAL FIX START ---
    
    # 1. Convert Output to Numpy (to match img_hr_np format)
    # Note: torch_to_np usually removes batch dim and handles normalization
    out_final_np = torch_to_np(output)
    img_hr_np = torch_to_np(img_hr_tensor)
    
    # 2. Get Dimensions (Channels, Height, Width)
    # Check if shapes match (3, H, W)
    h_sr, w_sr = out_final_np.shape[1], out_final_np.shape[2]
    h_hr, w_hr = img_hr_np.shape[1], img_hr_np.shape[2]
    
    # 3. Crop Ground Truth to match the SR Output size
    # We take the top-left crop which is standard for these datasets
    h_min = min(h_sr, h_hr)
    w_min = min(w_sr, w_hr)
    
    out_final_np = out_final_np[:, :h_min, :w_min]
    img_hr_np = img_hr_np[:, :h_min, :w_min]
    
    # --- CRITICAL FIX END ---

    # Calculate final metrics using the aligned numpy arrays
    final_psnr = peak_signal_noise_ratio(img_hr_np, out_final_np, data_range=1.0)
    print(f"Final PSNR: {final_psnr:.2f} dB")

    # Calculate SSIM
    # SSIM expects [H, W, C] for multichannel
    out_hwc = np.transpose(out_final_np, (1, 2, 0))    # [H,W,3]
    gt_hwc  = np.transpose(img_hr_np,  (1, 2, 0))      # [H,W,3]

    ssim_value = ssim(out_hwc, gt_hwc, channel_axis=2, data_range=1.0)
    print(f"Final SSIM: {ssim_value:.4f}")
    
    # Calculate LPIPS
    # LPIPS expects Tensor [Batch, 3, H, W] in range [-1, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert back to tensor for LPIPS and ensure shapes match
    sr_lpips = torch.from_numpy(out_final_np).unsqueeze(0).to(device)  # [1,3,H,W]
    hr_lpips = torch.from_numpy(img_hr_np).unsqueeze(0).to(device)      # [1,3,H,W]
    
    sr_lpips = (sr_lpips * 2 - 1)  # scale to [-1,1]
    hr_lpips = (hr_lpips * 2 - 1)
    
    lpips_value = loss_fn(sr_lpips, hr_lpips).item()
    print(f"Final LPIPS: {lpips_value:.4f}")
    
    # Log final images
    writer.add_image(f'Images/img{image_idx}_LR_input', torch.from_numpy(torch_to_np(img_lr_tensor)), image_idx)
    writer.add_image(f'Images/img{image_idx}_HR_ground_truth', torch.from_numpy(img_hr_np), image_idx)
    writer.add_image(f'Images/img{image_idx}_SR_output', torch.from_numpy(out_final_np), image_idx)

    return final_psnr, ssim_value, lpips_value

class Logger:
    """Simple logger that writes to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# --- RUN IT ---

# # Save Result
# output_pil = TF.to_pil_image(output_t.squeeze(0).cpu().clamp(0, 1))
# output_pil.save('result_swinir.png')
# print("Saved result_swinir.png")

def main():
    # Create single TensorBoard writer for entire dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'runs'
    run_dir = f'{log_dir}/SWINIR_{timestamp}'
    writer = SummaryWriter(run_dir)
    
    # Create text log file
    os.makedirs(run_dir, exist_ok=True)
    log_file = f'{run_dir}/training_log.txt'
    logger = Logger(log_file)
    sys.stdout = logger
    
    print(f"DIP Training started at {timestamp}")
    print(f"Logs will be saved to: {run_dir}")
    print(f"Text log: {log_file}\n")
    
    psnr_values = []
    ssim_values = []
    lpips_values = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create loss function for LPIPS
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    for i, (lr_image, hr_image) in enumerate(valid_loader):
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(valid_loader)}")
        print(f"{'='*60}")
        # print("lr",lr_image.shape)
        # print("hr", hr_image.shape)
        
                # Load Image
        img_lr = lr_image
        img_t = img_lr.to(DEVICE)
        img_hr = hr_image
        img_t_hr = img_hr.to(DEVICE)

        # Run Inference
        psnr, ssim_val, lpips_val= run_swinir(img_t, img_t_hr, loss_fn, i, writer)

        psnr_values.append(psnr)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)
        
        # Log per-image metrics
        writer.add_scalar('Dataset/PSNR_per_image', psnr, i)
        writer.add_scalar('Dataset/SSIM_per_image', ssim_val, i)
        writer.add_scalar('Dataset/LPIPS_per_image', lpips_val, i)

    # Calculate and log average metrics
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_lpips = sum(lpips_values) / len(lpips_values)
    
    print(f"\n{'='*60}")
    print(f"Average PSNR over validation dataset: {avg_psnr:.2f} dB")
    print(f"Average SSIM over validation dataset: {avg_ssim:.4f}")
    print(f"Average LPIPS over validation dataset: {avg_lpips:.4f}")
    print(f"{'='*60}")
    
    writer.add_scalar('Dataset/Average_PSNR', avg_psnr, 0)
    writer.add_scalar('Dataset/Average_SSIM', avg_ssim, 0)
    writer.add_scalar('Dataset/Average_LPIPS', avg_lpips, 0)
    writer.close()
    
    print(f"\nAll logs saved to {run_dir}")
    print(f"Text log saved to: {log_file}")
    print(f"View with: tensorboard --logdir={log_dir}")
    
    # Close logger and restore stdout
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()