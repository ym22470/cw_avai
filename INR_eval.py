# some imports
import numpy as np
import torch
import torch.nn as nn
import random
from scipy.ndimage import laplace, sobel
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
from dataloader import DIV2KDataset
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from utils.utils_DIP.denoising_utils import *
from models import downsampler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


PLOT = True
figsize = 8

def get_mgrid(sidelen1,sidelen2, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''

    if sidelen1 >= sidelen2:
      # use sidelen1 steps to generate the grid
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = sidelen1)])
      mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim = -1)
      # crop it along one axis to fit sidelen2
      minor = int((sidelen1 - sidelen2)/2)
      mgrid = mgrid[:,minor:sidelen2 + minor]

    if sidelen1 < sidelen2:
      tensors = tuple(dim * [torch.linspace(-1, 1, steps = sidelen2)])
      mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim = -1)

      minor = int((sidelen2 - sidelen1)/2)
      mgrid = mgrid[minor:sidelen1 + minor,:]

    # flatten the gird
    mgrid = mgrid.reshape(-1, dim)

    return mgrid


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
        # Initialize a linear layer with specified input and output features
        # 'bias' indicates whether to include a bias term
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
        # Task 1 TODO
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

def train_siren_for_one_image(lr_image, hr_image, writer, image_idx, num_iter=5000, batch_size=1, lr=1e-4
                              ):
    lr_image_VR = lr_image

    # Upsample LR image to match HR size
    import torch.nn.functional as F
    lr_image = F.interpolate(lr_image.unsqueeze(0), size=hr_image.shape[1:], mode='bicubic', align_corners=False).squeeze(0)

    # Convert tensors to numpy arrays
    lr_image_np = np.clip(torch_to_np(lr_image.unsqueeze(0)), 0, 1)


    # Get width and height
    width, height = lr_image_np.shape[2], lr_image_np.shape[1]
    print(f"Width: {width}, Height: {height}")

    # set the device to 'cuda' if available, otherwise 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize a SIREN model
    img_siren = Siren(in_features=2, out_features=3, hidden_features=256,
                    hidden_layers=3, outermost_linear=True)
    img_siren = img_siren.to(device)

    # Prepare LR→HR super-resolution data from DIV2K dataset
    # We already have lr_image_VR (low-res) and hr_image (high-res) from the dataset

    # Convert to proper tensors in [-1,1] range
    hr_tensor = hr_image.to(device)  # [C,H,W], already in [0,1]
    hr_tensor = (hr_tensor - 0.5) / 0.5  # normalize to [-1,1]

    lr_tensor = lr_image_VR.to(device)  # [C,h_lr,w_lr], already in [0,1]
    lr_tensor = (lr_tensor - 0.5) / 0.5  # normalize to [-1,1]

    C, H, W = hr_tensor.shape
    _, h_lr, w_lr = lr_tensor.shape

    print(f"HR shape: {hr_tensor.shape}, LR shape: {lr_tensor.shape}")
    print(f"Scale factor: ~{H/h_lr:.1f}x in height, ~{W/w_lr:.1f}x in width")


    # Super-resolution training loop: fit SIREN to map LR→HR
    # Strategy: Train on LR coordinates to predict HR pixel values

    # Build coordinate grids
    lr_coords = get_mgrid(h_lr, w_lr, dim=2).to(device)  # [h_lr*w_lr, 2]
    hr_coords = get_mgrid(H, W, dim=2).to(device)        # [H*W, 2]

    # Prepare pixel targets
    lr_pixels = lr_tensor.permute(1,2,0).reshape(-1, C)  # [h_lr*w_lr, C]
    hr_pixels = hr_tensor.permute(1,2,0).reshape(-1, C)  # [H*W, C]

    # Initialize SIREN for SR
    sr_siren = Siren(in_features=2, out_features=3, hidden_features=256,
                    hidden_layers=3, outermost_linear=True).to(device)

    optimizer = torch.optim.Adam(sr_siren.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    n_epochs = 5000
    print_every = 500

    # Training: fit SIREN on LR coords→LR pixels, then evaluate at HR coords
    for epoch in range(1, n_epochs+1):
        optimizer.zero_grad()

        # Predict RGB at LR coords (supervise on LR observations)
        pred_lr, _ = sr_siren(lr_coords)
        
        loss = criterion(pred_lr, lr_pixels)
        loss.backward()
        optimizer.step()

        # Log every iteration to TensorBoard with image-specific tag
        global_step = image_idx * n_epochs + epoch
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
        sr_image = sr_output.view(H, W, C).cpu()  # [H,W,C]
        sr_image = sr_image * 0.5 + 0.5  # denormalize to [0,1]
        
        # Also get LR prediction for comparison
        lr_pred, _ = sr_siren(lr_coords)
        lr_pred_image = lr_pred.view(h_lr, w_lr, C).cpu()
        lr_pred_image = lr_pred_image * 0.5 + 0.5
    # Calculate PSNR
    mse = torch.mean((sr_image - (hr_tensor.cpu().permute(1,2,0)*0.5+0.5))**2)
    psnr = 10 * torch.log10(1.0 / mse)
    print(f"\nFinal PSNR: {psnr.item():.2f} dB")
    
    # Log final images to TensorBoard with image index
    writer.add_image(f'Images/img{image_idx}_LR_input', lr_tensor.cpu()*0.5+0.5, image_idx)
    writer.add_image(f'Images/img{image_idx}_HR_ground_truth', hr_tensor.cpu()*0.5+0.5, image_idx)
    writer.add_image(f'Images/img{image_idx}_SR_output', sr_image.permute(2,0,1), image_idx)
    
    return psnr.item()


## Main training loop over validation dataset
def main():
    # Create single TensorBoard writer for entire dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'runs'
    writer = SummaryWriter(f'{log_dir}/siren_sr_dataset_{timestamp}')
    
    psnr_values = []
    
    for i, (lr_image, hr_image) in enumerate(valid_loader):
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(valid_loader)}")
        print(f"{'='*60}")
        psnr = train_siren_for_one_image(lr_image.squeeze(0), hr_image.squeeze(0), writer, i)
        psnr_values.append(psnr)
        
        # Log per-image PSNR to dataset-level metrics
        writer.add_scalar('Dataset/PSNR_per_image', psnr, i)

    # Calculate and log average PSNR over dataset
    avg_psnr = sum(psnr_values) / len(psnr_values)
    print(f"\n{'='*60}")
    print(f"Average PSNR over validation dataset: {avg_psnr:.2f} dB")
    print(f"{'='*60}")
    
    writer.add_scalar('Dataset/Average_PSNR', avg_psnr, 0)
    writer.close()
    print(f"\nAll logs saved to {writer.log_dir}")
    print(f"View with: tensorboard --logdir={log_dir}")

    
if __name__ == "__main__":
    main()
