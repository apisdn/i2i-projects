import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ImageDataset3to1, ImageDataset, CSVImageDataset
from model import Unet
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
import wandb
import numpy as np
import os
from PIL import Image

SAVE_DIR = "predictions3to1"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_INPUTS = 1
MODEL_NAME = "model3to1.pth"
RUN_NAME = "experiment3to1"

losses = []
val_losses = []
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device)

"""
Training function for the model
"""
def train(img_dataloader, model, opt, loss_fn, scaler):
    loop = tqdm(img_dataloader, leave=True)
    for idx, (x, y, imgidx) in enumerate(loop):
        #print(idx, x.shape, y.shape)
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        loss = loss_fn(preds, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())

        wandb.log({"Loss": loss.item()})

"""
Inference function for the model
"""
def infer(img_dataloader, model):
    model.eval()
    with torch.no_grad():
        for x,y,imgidx in img_dataloader:
            x = x.to(device)

            pred = model(x)

            datas = img_dataloader.dataset

            filename = datas.dataset.get_filenames(imgidx)

            pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred = (pred * 255).astype("uint8")

            filename = os.path.basename(filename)
            filename = filename.split(".")[0]
            filename = filename.split("_")[0]

            Image.fromarray(pred).save(os.path.join(SAVE_DIR, filename + "_fake.png"))
            wandb.log({"Validation Predictions": wandb.Image(os.path.join(SAVE_DIR, filename + "_fake.png"), caption=filename)})

"""
This main takes in a predict_only parameter, which is currently unimplemented
predict_only: bool, whether to only run the inference part of the code <<unimplemented>>
"""
def main(predict_only=False):
    # parameters
    in_chan = 3
    out_chan = 3
    learning_rate = 1e-4#1e-4
    batch_size = 5
    num_epochs = 200
    loss_fn = nn.CrossEntropyLoss()

    run = wandb.init(project="unet-translation-3to1",
                     job_type="train",
                     config={"learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs},
                     notes="Training a UNet model for image-to-image translation")

    model = Unet(in_channels=in_chan, out_channels=out_chan).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.GradScaler(device)

    if NUM_INPUTS == 3:
        ds = ImageDataset3to1(os.path.join("gainrangedataset","tir"), os.path.join("gainrangedataset","rgb","*"))
    else:
        #ds = ImageDataset(os.path.join("gainrangedataset","tir", "*_1.png"), os.path.join("gainrangedataset","rgb","*"))
        ds = CSVImageDataset(os.path.join('fake_therm_set','*.png'), "gainrangedataset/tir.csv", "gainrangedataset/rgb", "gainrangedataset/tir")
    
    training_set, validation_set = torch.utils.data.random_split(ds, [int(len(ds) * 0.7), len(ds) - int(len(ds) * 0.7)])

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False)

    # functionality to load a model?

    if predict_only:
        print("unimplemented")
        return
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        train(train_loader, model, opt, nn.MSELoss(), scaler)
        torch.save(model.state_dict(), MODEL_NAME)
        
    run.log_model(MODEL_NAME, name=RUN_NAME)

    print("Validation")
    model.eval()
    with torch.no_grad():
        for x, y, idx in val_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = loss_fn(preds, y)
            val_losses.append(loss.item())

    wandb.define_metric("Mean Validation Loss", summary="mean")
    wandb.log({"Mean Validation Loss": sum(val_losses) / len(val_losses)})

    print("Training complete")
    print("Mean validation loss: ", sum(val_losses) / len(val_losses))

    pkl.dump(train_loader, open("train_loader.pkl", "wb"))
    pkl.dump(val_loader, open("val_loader.pkl", "wb"))
    run.save("train_loader.pkl")
    run.save("val_loader.pkl")

    # and now for the validation set

    infer(val_loader, model)


if __name__ == "__main__":
    main(predict_only=False)