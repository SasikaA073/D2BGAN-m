"""
Training for D2BGAN (Low Light Image Enhancement)

Programmed by Sasika Amarasinghe <sasikapamith2016@gmail.com>
* 2024-03-12: Initial coding
"""

import torch
import sys
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

import config


from helper_transforms import IlluminationTransform, GeometricTransform, InverseGeometricTransform
from helper_transforms import gaussian_blur_tensor, get_grayscale_tensor, prewitt_edge_detection_tensor


def train_d2bgan(epoch, gen_X, gen_Y, disc_Xc1, disc_Xc2, disc_Xt, disc_Xe, disc_Yc1, disc_Yc2, disc_Yt, disc_Ye, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, dataloader):
    # get a progress bar
    loop = tqdm(dataloader, leave=True)

    # x = dark, y = bright
    for idx, (X, Y) in enumerate(loop):
        # change the device of images
        X_real = X.to(config.DEVICE)
        Y_real = Y.to(config.DEVICE)

        # Train Discriminators X and Y
        with torch.cuda.amp.autocast(): # to speed up training, dynamically autocast datatypes according to the context

            # DISCRIMINATOR X


            # generate dark image from bright image
            X_fake = gen_X(Y_real)

            X_g_real = GeometricTransform(X_real)
            Y_g_fake = gen_Y(X_g_real)
            Y_g_fake_inverse = InverseGeometricTransform(Y_g_fake)
 
            # pass though illumination transformed X real image thorugh generator X
            X_i_real = IlluminationTransform(X_real)
            Y_i_fake = gen_Y(X_i_real)


            ####################################### see whether dark discriminator can identify whether they are real or fake
            # Dark discriminators
            # colors

            # 2 blur images 
            Xc1_real = gaussian_blur_tensor(X_real, kernel_size = 3, sigma = config.BLUR_SIGMA_1)
            Xc1_fake = gaussian_blur_tensor(X_fake, kernel_size = 3, sigma = config.BLUR_SIGMA_1)
            
            Xc2_real = gaussian_blur_tensor(X_real, kernel_size = 3, sigma = config.BLUR_SIGMA_2)
            Xc2_fake = gaussian_blur_tensor(X_fake, kernel_size = 3, sigma = config.BLUR_SIGMA_2)

            D_Xc1_real = disc_Xc1(Xc1_real)
            D_Xc1_fake = disc_Xc1(Xc1_fake.detach())

            D_Xc2_real = disc_Xc2(Xc2_real)
            D_Xc2_fake = disc_Xc2(Xc2_fake.detach())

            # texture 
            Xt_real = get_grayscale_tensor(X_real)
            Xt_fake = get_grayscale_tensor(X_fake)
            
            D_Xt_real = disc_Xt(Xt_real)
            D_Xt_fake = disc_Xt(Xt_fake.detach())

            #edge
            Xe_real = prewitt_edge_detection_tensor(X_real)
            Xe_fake = prewitt_edge_detection_tensor(X_fake)
            
            D_Xe_real = disc_Xe(Xe_real)
            D_Xe_fake = disc_Xt(Xe_fake.detach())

            # See how many times dark discriminator made incorrect predictions
            D_Xc1_real_loss = mse(D_Xc1_real, torch.ones_like(D_Xc1_real)) # if real, target = 1
            D_Xc1_fake_loss = mse(D_Xc1_fake, torch.zeros_like(D_Xc1_fake)) # if fake, target = 0
             
            D_Xc2_real_loss = mse(D_Xc2_real, torch.ones_like(D_Xc2_real)) # if real, target = 1
            D_Xc2_fake_loss = mse(D_Xc2_fake, torch.zeros_like(D_Xc2_fake))
            
            D_Xt_real_loss = mse(D_Xt_real, torch.ones_like(D_Xt_real)) # if real, target = 1
            D_Xt_fake_loss = mse(D_Xt_fake, torch.zeros_like(D_Xt_fake))
            
            D_Xe_real_loss = mse(D_Xe_real, torch.ones_like(D_Xe_real)) # if real, target = 1
            D_Xe_fake_loss = mse(D_Xe_fake, torch.zeros_like(D_Xe_fake))
            
            # put it together - I normalized this loss value
            L_gan_X_Y = (D_Xc1_real_loss + D_Xc1_fake_loss + D_Xc2_real_loss + D_Xc2_fake_loss + D_Xt_real_loss + D_Xt_fake_loss + D_Xe_real_loss + D_Xe_fake_loss)/8
            ###########################################

            # DISCRIMINATOR Y
            # generate bright image from dark image
            Y_fake = gen_Y(X_real)

            # pass through geometric transformed X real through generator X
            X_g_real = GeometricTransform(X_real)
            Y_g_fake = gen_Y(X_g_real)
            Y_g_fake_inverse = InverseGeometricTransform(Y_g_fake)

            # pass though illumination transformed X real image thorugh generator X
            X_i_real = IlluminationTransform(X_real)
            Y_i_fake = gen_Y(X_i_real)

            #############################################################
            # see whether bright discriminator can identify whether they are real or fake

            # Here instead of one discriminator, they have used 3 discriminators for color, texture, edge
            # colors
            Yc1_real = gaussian_blur_tensor(Y_real, kernel_size = 3, sigma = config.BLUR_SIGMA_1)
            Yc1_fake = gaussian_blur_tensor(Y_fake, kernel_size = 3, sigma = config.BLUR_SIGMA_1)
            
            Yc2_real = gaussian_blur_tensor(Y_real, kernel_size = 3, sigma = config.BLUR_SIGMA_2)
            Yc2_fake = gaussian_blur_tensor(Y_fake, kernel_size = 3, sigma = config.BLUR_SIGMA_2)

            
            D_Yc1_real = disc_Yc1(Yc1_real)
            D_Yc1_fake = disc_Yc1(Yc1_fake.detach())
            
            D_Yc2_real = disc_Yc2(Yc2_real)
            D_Yc2_fake = disc_Yc2(Yc2_fake.detach())
            
            # texture
            Yt_real = get_grayscale_tensor(Y_real)
            Yt_fake = get_grayscale_tensor(Y_fake)
            
            D_Yt_real = disc_Yt(Yt_real)
            D_Yt_fake = disc_Yt(Yt_fake.detach())
            
            # edge
            Ye_real = prewitt_edge_detection_tensor(Y_real)
            Ye_fake = prewitt_edge_detection_tensor(Y_fake)
            
            D_Ye_real = disc_Ye(Ye_real)
            D_Ye_fake = disc_Ye(Ye_fake.detach())
            
            # See how many times bright discriminator made incorrect predictions
            D_Yc1_real_loss = mse(D_Yc1_real, torch.ones_like(D_Yc1_real)) # if real, target = 1
            D_Yc1_fake_loss = mse(D_Yc1_fake, torch.zeros_like(D_Yc1_fake)) # if fake, target = 0
            
            D_Yc2_real_loss = mse(D_Yc2_real, torch.ones_like(D_Yc2_real)) # if real, target = 1
            D_Yc2_fake_loss = mse(D_Yc2_fake, torch.zeros_like(D_Yc2_fake))
            
            D_Yt_real_loss = mse(D_Yt_real, torch.ones_like(D_Yt_real)) # if real, target = 1
            D_Yt_fake_loss = mse(D_Yt_fake, torch.zeros_like(D_Yt_fake))
            
            D_Ye_real_loss = mse(D_Ye_real, torch.ones_like(D_Ye_real)) # if real, target = 1
            D_Ye_fake_loss = mse(D_Ye_fake, torch.zeros_like(D_Ye_fake))
            
            # put it together - I normalized this loss value
            L_gan_Y_X = (D_Yc1_real_loss + D_Yc1_fake_loss + D_Yc2_real_loss + D_Yc2_fake_loss + D_Yt_real_loss + D_Yt_fake_loss + D_Ye_real_loss + D_Ye_fake_loss)/8
            #############################################################
            
            # put it together
            L_gan_total = (L_gan_X_Y + L_gan_Y_X)/2
            
            D_loss = L_gan_total
            
            
            

        # Performs a training step for discriminators
        opt_disc.zero_grad() # clears the gradients of all optimized parameters of the discriminator
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators X and Y
        with torch.cuda.amp.autocast():
            ##################################################################
            # Generate dark image using fake bright image
            X_recon = gen_X(Y_fake)

            # Generate bright image using fake dark image
            Y_recon = gen_Y(X_fake)


            cycle_Y_loss = L1(Y_real, Y_recon)
            cycle_X_loss = L1(X_real, X_recon)
            
            # Cycle Reconstruction Loss
            L_cyc = (cycle_Y_loss + cycle_X_loss)/2
            
            L_d2b_base = (L_gan_total + L_cyc)/2
            
            # Cyclic Consistency Loss
            L_cyc_con = (L1(Y_fake, Y_g_fake_inverse) + L1(Y_fake, Y_i_fake))/2
            
            L_d2bgan = (L_d2b_base + L_cyc_con)/2

            # add all together
            G_loss = L_d2bgan

        # Performs a training step for generators
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0 and epoch % 10 == 0:
            save_image(X_real*0.5+0.5, f"saved_images/real_dark_{epoch}_{idx}.png")
            save_image(Y_real*0.5+0.5, f"saved_images/real_bright_{epoch}_{idx}.png")
            save_image(X_fake*0.5+0.5, f"saved_images/fake_dark_{epoch}_{idx}.png")
            save_image(Y_fake*0.5+0.5, f"saved_images/fake_bright_{epoch}_{idx}.png")

