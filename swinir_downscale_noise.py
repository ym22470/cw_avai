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
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from models.network_swinir import SwinIR


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth' 
IMG_PATH = 'dataset/DIV2K_valid_LR_x8/0801x8.png'

use_gpu = True
noise_level = 0.5

if use_gpu:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.float32


tfs = transforms.Compose([
    transforms.ToTensor()
])

# Initialize Dataset
valid_dataset = DIV2KDataset(
    hr_dir="dataset/DIV2K_valid_HR/DIV2K_valid_HR",
    lr_dir="dataset/DIV2K_valid_LR_x8/DIV2K_valid_LR_x8",  
    transform=tfs
)

# Initialize DataLoader (Batch size 1 for validation/DIP)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

model = SwinIR(upscale=8,           
               in_chans=3, 
               img_size=48,           
               window_size=8, 
               img_range=1., 
               depths=[6, 6, 6, 6, 6, 6], 
               embed_dim=180, 
               num_heads=[6, 6, 6, 6, 6, 6], 
               mlp_ratio=2, 
               upsampler='pixelshuffle',
               resi_connection='1conv')

# Load the weights
pretrained_dict = torch.load(MODEL_PATH,weights_only=False)
param_key = 'params_ema' if 'params_ema' in pretrained_dict else 'params'
model.load_state_dict(pretrained_dict[param_key] if param_key in pretrained_dict else pretrained_dict)

model.eval()
model = model.to(DEVICE)

print("Model Loade")

def run_swinir_x16(img_lr_tensor, img_hr_tensor, loss_fn, image_idx, writer):
    _, _, h, w = img_lr_tensor.size()
    
    # Calculate padding needed
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    # Pad image
    img_lr_padded = torch.nn.functional.pad(img_lr_tensor, (0, pad_w, 0, pad_h), 'reflect')

    with torch.no_grad():
        # SwinIR x8 upscale
        output_x8 = model(img_lr_padded)
        # Unpad after x8
        output_x8 = output_x8[:, :, :h*8, :w*8]
        output = F.interpolate(output_x8, scale_factor=2, mode='bicubic', align_corners=False)

    # Convert Output to Numpy
    out_final_np = torch_to_np(output)
    img_hr_np = torch_to_np(img_hr_tensor)
    
    # Get Dimensions and align sizes
    h_sr, w_sr = out_final_np.shape[1], out_final_np.shape[2]
    h_hr, w_hr = img_hr_np.shape[1], img_hr_np.shape[2]
    
    h_min = min(h_sr, h_hr)
    w_min = min(w_sr, w_hr)
    
    out_final_np = out_final_np[:, :h_min, :w_min]
    img_hr_np = img_hr_np[:, :h_min, :w_min]
    print(f"Aligned SR shape###################################################: {out_final_np.shape}, GT shape: {img_hr_np.shape}")

    # Calculate final metrics
    final_psnr = peak_signal_noise_ratio(img_hr_np, out_final_np, data_range=1.0)
    print(f"Final PSNR: {final_psnr:.2f} dB")

    # Calculate SSIM
    out_hwc = np.transpose(out_final_np, (1, 2, 0))
    gt_hwc  = np.transpose(img_hr_np,  (1, 2, 0))
    ssim_value = ssim(out_hwc, gt_hwc, channel_axis=2, data_range=1.0)
    print(f"Final SSIM: {ssim_value:.4f}")
    
    # Calculate LPIPS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr_lpips = torch.from_numpy(out_final_np).unsqueeze(0).to(device)
    hr_lpips = torch.from_numpy(img_hr_np).unsqueeze(0).to(device)
    sr_lpips = (sr_lpips * 2 - 1)
    hr_lpips = (hr_lpips * 2 - 1)
    lpips_value = loss_fn(sr_lpips, hr_lpips).item()
    print(f"Final LPIPS: {lpips_value:.4f}")
    
    # Log final images
    writer.add_image(f'Images/img{image_idx}_LR_input', torch.from_numpy(torch_to_np(img_lr_tensor)), image_idx)
    writer.add_image(f'Images/img{image_idx}_HR_ground_truth', torch.from_numpy(img_hr_np), image_idx)
    writer.add_image(f'Images/img{image_idx}_SR_output', torch.from_numpy(out_final_np), image_idx)

    return final_psnr, ssim_value, lpips_value


# 3. Define the Inference Function
def run_swinir(img_lr_tensor, img_hr_tensor, loss_fn, image_idx, writer):
    """
    SwinIR requires image dimensions to be multiples of the window_size (8).
    We pad the image, run inference, and then unpad it.
    """
    _, _, h, w = img_lr_tensor.size()
    
    # Calculate padding needed
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    # Pad image
    img_lr_tensor = torch.nn.functional.pad(img_lr_tensor, (0, pad_w, 0, pad_h), 'reflect')

    with torch.no_grad():
        output = model(img_lr_tensor)

    # Unpad
    output = output[:, :, :h*8, :w*8]

    
    out_final_np = torch_to_np(output)
    img_hr_np = torch_to_np(img_hr_tensor)
    

    h_sr, w_sr = out_final_np.shape[1], out_final_np.shape[2]
    h_hr, w_hr = img_hr_np.shape[1], img_hr_np.shape[2]
    
    h_min = min(h_sr, h_hr)
    w_min = min(w_sr, w_hr)
    
    out_final_np = out_final_np[:, :h_min, :w_min]
    img_hr_np = img_hr_np[:, :h_min, :w_min]
    
    final_psnr = peak_signal_noise_ratio(img_hr_np, out_final_np, data_range=1.0)
    print(f"Final PSNR: {final_psnr:.2f} dB")

    # Calculate SSIM
    out_hwc = np.transpose(out_final_np, (1, 2, 0)) 
    gt_hwc  = np.transpose(img_hr_np,  (1, 2, 0))   

    ssim_value = ssim(out_hwc, gt_hwc, channel_axis=2, data_range=1.0)
    print(f"Final SSIM: {ssim_value:.4f}")
    
    # Calculate LPIPS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert back to tensor for LPIPS and ensure shapes match
    sr_lpips = torch.from_numpy(out_final_np).unsqueeze(0).to(device)
    hr_lpips = torch.from_numpy(img_hr_np).unsqueeze(0).to(device)    
    
    sr_lpips = (sr_lpips * 2 - 1) 
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

def add_awgn(x, sigma):
    """
    x: tensor in [0,1], shape [C,H,W]
    sigma: noise std in [0,1] scale
    """
    if sigma == 0:
        return x
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(0.0, 1.0)

def downsample(x, scale_factor=1/2):
    return F.interpolate(x, scale_factor=scale_factor, mode='bicubic', 
                         align_corners=False, antialias=True)


def main():
    # Create single TensorBoard writer for entire dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'runs'
    run_dir = f'{log_dir}/SWINIR)_downscaled_x16_{timestamp}'
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
        print(f"Processing down sacled image {i+1}/{len(valid_loader)}")
        print(f"{'='*60}")
        # print("lr",lr_image.shape)
        # print("hr", hr_image.shape)
        
                # Load Image - x8 LR from dataset, then x2 more downscale = x16 total
        img_lr = lr_image
        img_t = img_lr.to(DEVICE)

        img_t = downsample(img_t, scale_factor=1/2)  # x8 -> x16 LR

        print("After downscale x2, lr shape@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:", img_t.shape)
        img_hr = hr_image
        img_t_hr = img_hr.to(DEVICE)

        # Run Inference (SwinIR x8 + bicubic x2 for x16 total)
        psnr, ssim_val, lpips_val= run_swinir_x16(img_t, img_t_hr, loss_fn, i, writer)

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