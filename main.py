import os, time, subprocess

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import config
from dataset import LOLDataset
from utils import save_checkpoint, load_checkpoint, seed_everything
from discriminator_model import Discriminator
from generator_model import Generator
from train import train_d2bgan


def main():
    # Ensure repeatability
    seed_everything(42)
    
    ### Download dataset 
    # If dataset is not downloaded, download it
    if not os.path.exists(config.DARK_DIR):
        url = config.DATASET_URL 
        output_file = config.DATASET_ZIP_NAME
        
        # Run wget command using subprocess to download the dataset
        subprocess.run(['wget', url])

        # Unzip the downloaded file
        subprocess.run(['unzip', output_file])

        # Remove the downloaded zip file
        os.remove(output_file)
        
        # Run wget command using subprocess
        subprocess.run(['wget', '--no-check-certificate', url, '-O', output_file])

    ### Make dataloaders
    
    
    ### Make Discriminators
    dark_c1_discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    dark_c2_discriminator = Discriminator(in_channels=3).to(config.DEVICE)  # how to differentiate the following 4???
    dark_t_discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    dark_e_discriminator = Discriminator(in_channels=3).to(config.DEVICE)

    bright_c1_discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    bright_c2_discriminator = Discriminator(in_channels=3).to(config.DEVICE)  # how to differentiate the following 4???
    bright_t_discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    bright_e_discriminator = Discriminator(in_channels=3).to(config.DEVICE)

    #########################################################

    dark_generator = Generator(in_channels=3, out_channels=64).to(config.DEVICE)
    bright_generator = Generator(in_channels=3, out_channels=64).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(dark_c1_discriminator.parameters())
        + list(dark_c2_discriminator.parameters())
        + list(dark_t_discriminator.parameters())
        + list(dark_e_discriminator.parameters())
        + list(bright_c1_discriminator.parameters())
        + list(bright_c2_discriminator.parameters())
        + list(bright_t_discriminator.parameters())
        + list(bright_e_discriminator.parameters())
        ,
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999), # beta1 = 0.5, beta2 = 0.999 described as in the paper
    )

    opt_gen = optim.Adam(
        list(dark_generator.parameters()) + list(bright_generator.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999), # beta1 = 0.5, beta2 = 0.999 described as in the paper
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    # IF models are available load,
    
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_DARK, dark_generator, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_BRIGHT, bright_generator, opt_gen, config.LEARNING_RATE,
        )
        
        # Load checkpoints for dark discriminators
        load_checkpoint(
            config.CHECKPOINT_CRITIC_DARK_C1, dark_c1_discriminator, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_DARK_C2, dark_c2_discriminator, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_DARK_T, dark_t_discriminator, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_DARK_E, dark_e_discriminator, opt_disc, config.LEARNING_RATE
        )

        
        # Load checkpoints for bright discriminators
        load_checkpoint(
            config.CHECKPOINT_CRITIC_BRIGHT_C1, bright_c1_discriminator, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_BRIGHT_C2, bright_c2_discriminator, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_BRIGHT_T, bright_t_discriminator, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_BRIGHT_E, bright_e_discriminator, opt_disc, config.LEARNING_RATE
        )



    dataset = LOLDataset(
        root_dark= config.TRAIN_DIR + "low",
        root_bright= config.TRAIN_DIR + "high",
        transform= config.train_transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size= config.BATCH_SIZE,
        # shuffle=True,
        shuffle=False,
        num_workers=config.NUM_WORKERS,  # what does this do?
        # pin_memory=True, # what does this do?
        pin_memory=False
    )

    generator_scaler = torch.cuda.amp.GradScaler()
    discriminator_scaler = torch.cuda.amp.GradScaler()

    
    ### TRAIN MODEL 
    for epoch in range(config.NUM_EPOCHS):
    

        epoch_start_time = time.time()  # Timer for entire epoch


        train_stats = train_d2bgan(epoch, 
                                   dark_generator,
                                   bright_generator, 
                                   
                                   dark_c1_discriminator, 
                                   dark_c2_discriminator, 
                                   dark_t_discriminator,
                                   dark_e_discriminator, 
                                   
                                    bright_c1_discriminator,
                                    bright_c2_discriminator, 
                                    bright_t_discriminator, 
                                    bright_e_discriminator, 
                                    
                                    opt_disc, 
                                    opt_gen, 
                                    L1, 
                                    mse, 
                                    
                                    discriminator_scaler, 
                                    generator_scaler, 
                                    dataloader)
   
        if config.SAVE_MODEL and (epoch % config.SAVE_EPOCH_FREQ == 0):  # Fixed the condition
            save_checkpoint(dark_generator, opt_gen, filename=config.MODEL_PATH + config.CHECKPOINT_GEN_DARK)
            save_checkpoint(bright_generator, opt_gen, filename=config.MODEL_PATH + config.CHECKPOINT_GEN_BRIGHT)
            
            
            # Save checkpoints for dark domain
            save_checkpoint(dark_c1_discriminator, opt_disc, filename=config.MODEL_PATH + config.CHECKPOINT_CRITIC_DARK_C1)
            save_checkpoint(dark_c2_discriminator, opt_disc, filename=config.MODEL_PATH + config.CHECKPOINT_CRITIC_DARK_C2)
            save_checkpoint(dark_t_discriminator, opt_disc, filename=config.MODEL_PATH + config.CHECKPOINT_CRITIC_DARK_T)
            save_checkpoint(dark_e_discriminator, opt_disc, filename=config.MODEL_PATH + config.CHECKPOINT_CRITIC_DARK_E)

            # Save checkpoints for bright domain
            save_checkpoint(bright_c1_discriminator, opt_disc, filename=config.MODEL_PATH + config.CHECKPOINT_CRITIC_BRIGHT_C1)
            save_checkpoint(bright_c2_discriminator, opt_disc, filename=config.MODEL_PATH + config.CHECKPOINT_CRITIC_BRIGHT_C2)
            save_checkpoint(bright_t_discriminator, opt_disc, filename=config.MODEL_PATH + config.CHECKPOINT_CRITIC_BRIGHT_T)
            save_checkpoint(bright_e_discriminator, opt_disc, filename=config.MODEL_PATH + config.CHECKPOINT_CRITIC_BRIGHT_E)
            

        epoch_time_taken = time.time() - epoch_start_time
        print(f"Epoch {epoch} / Time Taken: {epoch_time_taken:.2f} sec\n")

        
    
if __name__ == "__main__":
    main()