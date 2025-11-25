import numpy as np
import torch.optim
import random
from dataloader import DIV2KDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from models import *
from utils.utils_DIP.denoising_utils import *
from skimage.metrics import peak_signal_noise_ratio

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

use_gpu = True

if use_gpu:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.float32

PLOT = True

INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

#training parameters
reg_noise_std = 1./30.
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 500
exp_weight=0.99

num_iter = 5000
input_depth = 3
figsize = 32
random_noise = False


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

# Initialize DataLoader (Batch size 1 for validation/DIP)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

# 1. Get the total number of images
total_images = len(valid_dataset)

# 2. Pick a random index
random_idx = random.randint(0, total_images - 1)

# 3. Retrieve the image pair directly
lr_image, hr_image = valid_dataset[29]

print(f"Selected Image Index: {random_idx}")

# Upsample LR image to match HR size
import torch.nn.functional as F
lr_image = F.interpolate(lr_image.unsqueeze(0), size=hr_image.shape[1:], mode='bicubic', align_corners=False).squeeze(0)

# Convert tensors to numpy arrays
img_noisy_np = np.clip(torch_to_np(lr_image.unsqueeze(0)), 0, 1)
img_np = np.clip(torch_to_np(hr_image.unsqueeze(0)), 0, 1)

# Crop to be divisible by 32
factor = 32
h, w = img_noisy_np.shape[1], img_noisy_np.shape[2]
new_h = h - h % factor
new_w = w - w % factor
img_noisy_np = img_noisy_np[:, :new_h, :new_w]
img_np = img_np[:, :new_h, :new_w]

if PLOT:
    plot_image_grid([img_np, img_noisy_np], 4, figsize)



##############################################################
###############Network structure##############################
##############################################################
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

# whether to generate random noise or noisy image
# if random_noise:
#     net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
# else:
net_input = np_to_torch(img_noisy_np).type(dtype)

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()])
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
img_torch = np_to_torch(img_np).type(dtype)


"""Start Training"""
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
#smoothing image
out_avg = net_input_saved

#without smoothing
#out_avg = None

last_net = None
psrn_noisy_last = 0

i = 0
def closure():

    global i, out_avg, psrn_noisy_last, last_net, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    #total_loss = mse(out, img_noisy_torch)
    total_loss = mse(out, img_torch)
    total_loss.backward()

    #evaluation with psrn
    psrn_noisy = peak_signal_noise_ratio(img_noisy_np, out.detach().cpu().numpy()[0])
    psrn_gt    = peak_signal_noise_ratio(img_np, out.detach().cpu().numpy()[0])
    psrn_gt_sm = peak_signal_noise_ratio(img_np, out_avg.detach().cpu().numpy()[0])

    if  PLOT and i % 10 == 0:
         print ('Iteration: ', i, ' Loss: ', total_loss.item(), ' PSRN_gt: ', psrn_gt, ' PSNR_gt_sm: ', psrn_gt_sm)
    #print ('Iteration %05d    Loss %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_gt, psrn_gt_sm), '\r', end='')
    if  PLOT and i % show_every == 0:
        #out_np = torch_to_np(out)
        plot_image_grid([np.clip(img_np, 0, 1),
                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=2)



    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy

    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


out_np = torch_to_np(net(net_input))
q = plot_image_grid([img_noisy_np, np.clip(out_np, 0, 1), img_np], factor=figsize)