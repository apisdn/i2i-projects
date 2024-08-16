import time
from dataset import ImageDataset3to1
from model import Pix2Pix
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import torch
import pickle as pkl
from PIL import Image

SAVE_DIR = "predictionsPIX2PIX"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_INPUTS = 3
MODEL_NAME = "model_current_"
RUN_NAME = "pix2pix-taylorsversion"

def train(img_dataloader, model):
    loop = tqdm(img_dataloader, leave=True)
    for idx, (x, y, imgidx) in enumerate(loop):
        model.set_input(y, x) # set backwards because i want to go from RGB to TIR
        model.optimize_parameters()

        losses = model.get_current_losses()
        wandb.log({'G_GAN': losses['G_GAN'], 'G_L1': losses['G_L1'], 'D_real': losses['D_real'], 'D_fake': losses['D_fake']})

        loop.set_postfix(loss=model.loss_G.item())

def validate(img_dataloader, model):
    model.netG.to(model.device)

    loop = tqdm(img_dataloader, leave=True)
    for idx, (x, y, imgidx) in enumerate(loop):
        model.set_input(y, x) # set backwards because i want to go from RGB to TIR
        model.test()

        pred = model.fake

        datas = img_dataloader.dataset

        filename = datas.dataset.get_filenames(imgidx)

        pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pred = (pred * 255).astype("uint8")

        filename = os.path.basename(filename)
        filename = filename.split(".")[0]
        filename = filename.split("_")[0]

        img = Image.fromarray(pred)

        channel1, channel2, channel3 = img.split()
        
        channel1.save(os.path.join(SAVE_DIR, filename + "_0_fake.png"))
        channel2.save(os.path.join(SAVE_DIR, filename + "_1_fake.png"))
        channel3.save(os.path.join(SAVE_DIR, filename + "_2_fake.png"))
        wandb.log({"Validation Predictions 0": wandb.Image(os.path.join(SAVE_DIR, filename + "_0_fake.png"), caption=filename)})
        wandb.log({"Validation Predictions 1": wandb.Image(os.path.join(SAVE_DIR, filename + "_1_fake.png"), caption=filename)})
        wandb.log({"Validation Predictions 2": wandb.Image(os.path.join(SAVE_DIR, filename + "_2_fake.png"), caption=filename)})

        # save the result, splitting each channel into a black and white image


if __name__ == '__main__':
    # parameters
    in_chan = 3
    out_chan = 3
    learning_rate = 1e-4#1e-4
    batch_size = 5
    num_epochs = 100

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

    pkl.dump(train_loader, open("train_loader.pkl", "wb"))
    pkl.dump(val_loader, open("val_loader.pkl", "wb"))
    run.save("train_loader.pkl")
    run.save("val_loader.pkl")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        model.netG.to(model.device)
        model.netD.to(model.device)

        train(train_loader, model)
        
        model.save_networks(epoch)
        
    run.log_model(MODEL_NAME + 'D.pth', name=RUN_NAME)
    run.log_model(MODEL_NAME + 'G.pth', name=RUN_NAME)
    
    print("Training complete")

    print("Validation:")
    validate(val_loader, model)



        