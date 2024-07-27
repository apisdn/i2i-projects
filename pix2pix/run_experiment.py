import time
from dataset import ImageDataset3to1
from model import Pix2Pix
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import torch

def train(img_dataloader, model):
    loop = tqdm(img_dataloader, leave=True)
    for idx, (x, y, imgidx) in enumerate(loop):
        model.set_input(x, y)
        model.optimize_parameters()

        losses = model.get_current_losses()
        wandb.log({'G_GAN': losses['G_GAN'], 'G_L1': losses['G_L1'], 'D_real': losses['D_real'], 'D_fake': losses['D_fake']})

        loop.set_postfix(loss=model.loss_G.item())


if __name__ == '__main__':
    RUN_NAME = "pix2pix-taylorsversion"

    # parameters
    in_chan = 3
    out_chan = 3
    learning_rate = 1e-4#1e-4
    batch_size = 5
    num_epochs = 5

    run = wandb.init(project="pix2pix-taylorsversion",
                     job_type="train",
                     config={"learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs},
                     notes="Training a Pix2Pix model for image-to-image translation")
    
    model = Pix2Pix(in_channels=in_chan, out_channels=out_chan)
    model.setup()

    ds = ImageDataset3to1(os.path.join("gainrangedataset","tir"), os.path.join("gainrangedataset","rgb","*"))
    training_set, validation_set = torch.utils.data.random_split(ds, [int(len(ds) * 0.7), len(ds) - int(len(ds) * 0.7)])

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        train(train_loader, model)
        if epoch+1 % 10 == 0:
            torch.save(model.state_dict(), f"pix2pix_{epoch}.pth")
        torch.save(model.state_dict(), "pix2pix_current.pth")
        
    run.log_model("pix2pix_current.pth", name=RUN_NAME)


        