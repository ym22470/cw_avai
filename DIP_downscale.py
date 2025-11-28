import numpy as np
import sys
import os
from datetime import datetime
import torch.nn.functional as F
import lpips
from dataloader import DIV2KDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import *
from utils.utils_DIP.denoising_utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

use_gpu = True

MAX_Images = 10

if use_gpu:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.float32

INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

#training parameters
reg_noise_std = 1./30.
LR = 0.05

OPTIMIZER='adam' # 'LBFGS'
show_every = 500
exp_weight=0.99

num_iter = 2000
input_depth = 32
figsize = 32
random_noise = True

INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'


# Define transforms (convert to tensor)
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

net = get_net(
    input_depth, 
    'skip', 
    pad='reflection', 
    upsample_mode='bilinear',
    skip_n33d=32,
    skip_n33u=32,  
    skip_n11=4,
    num_scales=5
).type(dtype)

def downsample(x, scale_factor=1/2):
    return F.interpolate(x, scale_factor=scale_factor, mode='bicubic', 
                         align_corners=False, antialias=True)

def add_awgn(x, sigma):
    """
    x: tensor in [0,1], shape [C,H,W]
    sigma: noise std in [0,1] scale
    """
    if sigma == 0:
        return x
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(0.0, 1.0)


def train_DIP_for_one_image(net, loss_fn,lr_image, hr_image, writer, image_idx, num_iter=num_iter, reg_noise_std = 0.05, factor=16, print_every=500):
    lr_image_VR = lr_image.unsqueeze(0).type(dtype)  # Add batch dimension and convert to dtype


    # Convert tensors to numpy arrays
    img_np = np.clip(torch_to_np(hr_image.unsqueeze(0)), 0, 1)

    # Setup noisy image
    net_input = get_noise(input_depth, INPUT, (hr_image.size(1), hr_image.size(2))).type(dtype).detach()

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()])
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    import torch.nn.functional as F

    # 1. Define the Downsampler (The "Physics" of the problem)
  

    """Start Training"""
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    out_avg = None  # start with no average; let the first 'out' define it

    last_net = None
    psrn_noisy_last = 0

    i = 0
    def closure():

        nonlocal i, out_avg, psrn_noisy_last, last_net, net_input

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        lr_out = downsample(out, scale_factor=1/factor)

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)


        #total_loss = mse(out, img_noisy_torch)
        total_loss = mse(lr_out, lr_image_VR)
        total_loss.backward()

        #evaluation with psrn
        psrn_gt    = peak_signal_noise_ratio(img_np, out.detach().cpu().numpy()[0])
        psrn_gt_sm = peak_signal_noise_ratio(img_np, out_avg.detach().cpu().numpy()[0])
        
        # Print progress less frequently - every 100 iterations instead of 10
        if i % 100 == 0:
            print ('Iteration: ', i, ' Loss: ', total_loss.item(), ' PSRN_gt: ', psrn_gt, ' PSNR_gt_sm: ', psrn_gt_sm)

        # Log to TensorBoard less frequently - every 100 iterations instead of every iteration
        if i % 100 == 0:
            writer.add_scalar(f'Loss/train_img{image_idx}', total_loss.item(), i)
            writer.add_scalar(f'Metrics/PSNR_img{image_idx}', psrn_gt, i)
            writer.add_scalar(f'Metrics/PSNR_smoothed_img{image_idx}', psrn_gt_sm, i)

        i += 1
        return total_loss


    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)
    
    # Final evaluation
    net.eval()
    with torch.no_grad():
        out_final = net(net_input)
        out_avg_final = out_avg if out_avg is not None else out_final
    
    out_np = torch_to_np(out_final)
    out_avg_np = torch_to_np(out_avg_final)
    
    # Calculate final metrics
    final_psnr = peak_signal_noise_ratio(img_np, out_avg_np)
    print(f"\nFinal PSNR (smoothed): {final_psnr:.2f} dB")
    
    # Calculate SSIM
    out_chw = out_avg_np                          # [3,H,W]
    gt_chw  = img_np                          # [3,H,W]

    out_hwc = np.transpose(out_chw, (1, 2, 0))    # [H,W,3]
    gt_hwc  = np.transpose(gt_chw,  (1, 2, 0))    # [H,W,3]

    ssim_value = ssim(out_hwc, gt_hwc,
                    channel_axis=2, data_range=1.0)
    print(f"Final SSIM: {ssim_value:.4f}")
    
    # Calculate LPIPS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = loss_fn.to(device)
    sr_lpips = torch.from_numpy(out_avg_np).to(device)  # [1,3,H,W]
    hr_lpips = torch.from_numpy(img_np).to(device)      # [1,3,H,W]
    sr_lpips = (sr_lpips * 2 - 1)  # scale to [-1,1]
    hr_lpips = (hr_lpips * 2 - 1)
    lpips_value = loss_fn(sr_lpips, hr_lpips).item()
    print(f"Final LPIPS: {lpips_value:.4f}")
    
    # Log final images
    writer.add_image(f'Images/img{image_idx}_LR_input', torch.from_numpy(lr_image.numpy()).squeeze(0), image_idx)
    writer.add_image(f'Images/img{image_idx}_HR_ground_truth', torch.from_numpy(img_np), image_idx)
    writer.add_image(f'Images/img{image_idx}_SR_output', torch.from_numpy(out_avg_np), image_idx)
    
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


def main():
    # Create single TensorBoard writer for entire dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'runs'
    run_dir = f'{log_dir}/dip_sigma_downscale_{timestamp}'
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
        if i >= MAX_Images:
            break
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(valid_loader)}")
        print(f"{'='*60}")
        
        # Reinitialize network weights for each image
        net = get_net(
        input_depth, 
        'skip', 
        pad='reflection', 
        upsample_mode='bilinear',
        skip_n33d=32,
        skip_n33u=32,  
        skip_n11=4,
        num_scales=5
        ).type(dtype)

        lr_noisy = downsample(lr_image, scale_factor=1/2)
        
        psnr, ssim_val, lpips_val = train_DIP_for_one_image(
            net, loss_fn, lr_noisy, hr_image.squeeze(0), writer, i, num_iter=num_iter
        )
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
