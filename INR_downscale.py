# some imports
import numpy as np
import torch
import os
import sys
import torch.nn as nn
import lpips 
from scipy.ndimage import laplace, sobel
from torch.utils.data import DataLoader, Dataset
from dataloader import DIV2KDataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from utils.utils_DIP.denoising_utils import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

MAX_Images = 10


def get_mgrid(H, W, dim=2):
    ys = torch.linspace(-1, 1, steps=H)
    xs = torch.linspace(-1, 1, steps=W)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
    return grid.reshape(-1, dim)


def image_to_tensor(img):
    transform = Compose([
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class SineLayer(nn.Module):

    """ Linear layer followed by the sine activation

    If `is_first == True`, then it represents the first layer of the network.
    In this case, omega_0 is a frequency factor, which simply multiplies the activations before the nonlinearity.
    Note that it influences the initialization scheme.

    If `is_first == False`, then the weights will be divided by omega_0 so as to keep the magnitude of activations constant,
    but boost gradients to the weight matrix.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    # initialize weights uniformly
    def init_weights(self):
        # diasble gradient calculation in initialization
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        # 1. pass input through linear layer (self.linear layer performs the linear transformation on the input)
        x = self.linear(input)

        # 2. scale the output of the linear transformation by the frequency factor
        x = x * self.omega_0

        # 3. apply sine activation
        x = torch.sin(x)

        return x

class Siren(nn.Module):
    """ SIREN architecture """

    def __init__(self, in_features, out_features, hidden_features=256, hidden_layers=3, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        # add the first layer
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        # append hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))


        if outermost_linear:
            # add a final Linear layer
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad(): # weights intialization
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            # otherwise, add a SineLayer
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net) # sequential wrapper of SineLayer and Linear

    def forward(self, coords):
        # coords represents the 2D pixel coordinates
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords
    

# Image Fitting Dataloader
class ImageData(Dataset):
    def __init__(self, img):
        super().__init__()

        # convert the image to a tensor with transformations
        img = image_to_tensor(img)

        self.pixels = img.permute(1, 2, 0).reshape(-1, 3) # pixel values of the org img

        # create a grid of coordinates for the image
        self.coords = get_mgrid(img.shape[1], img.shape[2], 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels
    
# Define transforms (convert to tensor)
tfs = transforms.Compose([
    transforms.ToTensor()
])

# Initialize Dataset
valid_dataset = DIV2KDataset(
    hr_dir="dataset/DIV2K_valid_HR/DIV2K_valid_HR",
    lr_dir="dataset/DIV2K_valid_LR_x8/DIV2K_valid_LR_x8",  # Check your unzipped folder name
    transform=tfs
)

valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

def train_siren_for_one_image(lr_image, hr_image, writer, image_idx, num_iter=5000, print_every=500):
    lr_image_VR = lr_image



    # set the device to 'cuda' if available, otherwise 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to proper tensors in [-1,1] range
    hr_tensor = hr_image.to(device) 
    hr_tensor = (hr_tensor - 0.5) / 0.5 

    lr_tensor = lr_image_VR.to(device)  
    lr_tensor = (lr_tensor - 0.5) / 0.5  

    C, H, W = hr_tensor.shape
    _, h_lr, w_lr = lr_tensor.shape

    print(f"HR shape: {hr_tensor.shape}, LR shape: {lr_tensor.shape}")
    print(f"Scale factor: ~{H/h_lr:.1f}x in height, ~{W/w_lr:.1f}x in width")


    # Super-resolution training loop

    # Build coordinate grids
    lr_coords = get_mgrid(h_lr, w_lr, dim=2).to(device) 
    hr_coords = get_mgrid(H, W, dim=2).to(device)        
    # Prepare pixel targets
    lr_pixels = lr_tensor.permute(1,2,0).reshape(-1, C) 
    hr_pixels = hr_tensor.permute(1,2,0).reshape(-1, C) 

    # Initialize SIREN for SR
    sr_siren = Siren(in_features=2, out_features=3, hidden_features=256,
                    hidden_layers=3, outermost_linear=True).to(device)

    optimizer = torch.optim.Adam(sr_siren.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    n_epochs = num_iter
    print_every = print_every

    # Training: fit SIREN on LR coords→LR pixels, then evaluate at HR coords
    for epoch in range(1, n_epochs+1):
        optimizer.zero_grad()

        # Predict RGB at LR coords (supervise on LR observations)
        pred_lr, _ = sr_siren(lr_coords)
        
        loss = criterion(pred_lr, lr_pixels)
        loss.backward()
        optimizer.step()

        # Log every iteration to TensorBoard with image-specific tag
        writer.add_scalar(f'Loss/LR_train_img{image_idx}', loss.item(), epoch)

        if epoch % print_every == 0 or epoch == 1:
            with torch.no_grad():
                # Also evaluate on HR coords to see SR quality
                pred_hr, _ = sr_siren(hr_coords)
                hr_loss = criterion(pred_hr, hr_pixels)
                
                # Calculate PSNR during training
                sr_image_temp = pred_hr.view(H, W, C).cpu() * 0.5 + 0.5
                mse_temp = torch.mean((sr_image_temp - (hr_tensor.cpu().permute(1,2,0)*0.5+0.5))**2)
                psnr_temp = 10 * torch.log10(1.0 / mse_temp)
                
                # Log to TensorBoard with image-specific tag
                writer.add_scalar(f'Loss/HR_eval_img{image_idx}', hr_loss.item(), epoch)
                writer.add_scalar(f'Metrics/PSNR_img{image_idx}', psnr_temp.item(), epoch)
                
            print(f"Epoch {epoch}/{n_epochs}, LR Loss: {loss.item():.6f}, HR Loss: {hr_loss.item():.6f}, PSNR: {psnr_temp.item():.2f} dB")

    sr_siren.eval()
    # Visualize super-resolution results
    with torch.no_grad():
        # Generate SR image by evaluating SIREN at HR coordinates
        sr_output, _ = sr_siren(hr_coords)
        sr_image = sr_output.view(H, W, C).cpu() 
        sr_image = sr_image * 0.5 + 0.5  # denormalize to [0,1]
        
        # Also get LR prediction for comparison
        lr_pred, _ = sr_siren(lr_coords)
        lr_pred_image = lr_pred.view(h_lr, w_lr, C).cpu()
        lr_pred_image = lr_pred_image * 0.5 + 0.5


    # Calculate PSNR
    mse = torch.mean((sr_image - (hr_tensor.cpu().permute(1,2,0)*0.5+0.5))**2)
    psnr = 10 * torch.log10(1.0 / mse)
    print(f"\nFinal PSNR: {psnr.item():.2f} dB")

    # Calculate SSIM
    # Denormalise HR back to [0,1]
    hr_img = (hr_tensor * 0.5 + 0.5).clamp(0, 1)    # Scale back to [0,1]
    hr_img = hr_img.permute(1, 2, 0).cpu() 
    sr_np = sr_image.cpu().numpy()
    hr_np = hr_img.numpy()
    ssim_value = ssim(sr_np, hr_np, channel_axis=2, data_range=1.0)
    print(f"Final SSIM: {ssim_value:.4f}")

    # Calculate LPIPS
    loss_fn = lpips.LPIPS(net='alex').to(device)
    hr_img = (hr_tensor * 0.5 + 0.5).clamp(0, 1)  
    sr_lpips = sr_image.permute(2, 0, 1).unsqueeze(0)   
    hr_lpips = hr_img.unsqueeze(0)                     
    sr_lpips = (sr_lpips * 2 - 1).to(device)  
    # scale to [-1,1]
    hr_lpips = (hr_lpips * 2 - 1).to(device)
    lpips_value = loss_fn(sr_lpips, hr_lpips).item()
    print(f"Final LPIPS: {lpips_value:.4f}")

    # Log final images to TensorBoard with image index
    writer.add_image(f'Images/img{image_idx}_LR_input', lr_tensor.cpu()*0.5+0.5, image_idx)
    writer.add_image(f'Images/img{image_idx}_HR_ground_truth', hr_tensor.cpu()*0.5+0.5, image_idx)
    writer.add_image(f'Images/img{image_idx}_SR_output', sr_image.permute(2,0,1), image_idx)
    
    return psnr.item(), ssim_value, lpips_value


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
import torch.nn.functional as F
def downsample(x, scale_factor=1/8):
    return F.interpolate(x, scale_factor=scale_factor, mode='bicubic', 
                         align_corners=False, antialias=True)
## Main training loop over validation dataset
def main():
    # Create single TensorBoard writer for entire dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'runs'
    run_dir = f'{log_dir}/siren_downscale_dataset_{timestamp}'
    writer = SummaryWriter(run_dir)
    
    # Create text log file
    os.makedirs(run_dir, exist_ok=True)
    log_file = f'{run_dir}/training_log.txt'
    logger = Logger(log_file)
    sys.stdout = logger
    
    print(f"Training started at {timestamp}")
    print(f"Logs will be saved to: {run_dir}")
    print(f"Text log: {log_file}\n")
    
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    for i, (lr_image, hr_image) in enumerate(valid_loader):
        if i >= MAX_Images:
            break
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(valid_loader)}")
        print(f"{'='*60}")
        
        # further downsample by 2x
        lr_image = downsample(lr_image, scale_factor=1/2)

        psnr, ssim_value, lpips_value = train_siren_for_one_image(lr_image.squeeze(0), hr_image.squeeze(0), writer, i)
        psnr_values.append(psnr)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)
        
        # Log per-image PSNR, SSIM, and LPIPS to dataset-level metrics
        writer.add_scalar('Dataset/PSNR_per_image', psnr, i)
        writer.add_scalar('Dataset/SSIM_per_image', ssim_value, i)
        writer.add_scalar('Dataset/LPIPS_per_image', lpips_value, i)

    # Calculate and log average PSNR, SSIM, and LPIPS over dataset
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_lpips = sum(lpips_values) / len(lpips_values)
    print(f"\n{'='*60}")
    print(f"Average PSNR over validation dataset: {avg_psnr:.2f} dB")
    print(f"Average SSIM over validation dataset: {avg_ssim:.4f}")
    print(f"Average LPIPS over validation dataset: {avg_lpips:.4f}")
    print(f"{'='*60}")
    
    writer.add_scalar('Dataset/Average_PSNR', avg_psnr, 0)
    writer.add_scalar("Dataset/Average_SSIM", avg_ssim, 0)
    writer.add_scalar("Dataset/Average_LPIPS", avg_lpips, 0)
    writer.close()
    print(f"\nAll logs saved to {run_dir}")
    print(f"Text log saved to: {log_file}")
    print(f"View with: tensorboard --logdir={log_dir}")
    
    # Close logger and restore stdout
    sys.stdout = logger.terminal
    logger.close()

    
if __name__ == "__main__":
    main()
