import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ImageDataset
from model import Unet
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
import wandb
import numpy as np

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
This main takes in a predict_only parameter, which is currently unimplemented
predict_only: bool, whether to only run the inference part of the code <<unimplemented>>
"""
def main(predict_only=False):
    RUN_NAME = "test2"

    # parameters
    in_chan = 3
    out_chan = 3
    learning_rate = 1e-4#1e-4
    batch_size = 5
    num_epochs = 5
    loss_fn = nn.CrossEntropyLoss()

    run = wandb.init(project="unet-translation-test",
                     job_type="train",
                     config={"learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs},
                     notes="Training a UNet model for image-to-image translation")

    model = Unet(in_channels=in_chan, out_channels=out_chan).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.GradScaler(device)

    ds = ImageDataset("real_rgb/rgb/*", "real_rgb/tir/*")
    training_set, validation_set = torch.utils.data.random_split(ds, [int(len(ds) * 0.8), len(ds) - int(len(ds) * 0.8)])

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    # functionality to load a model?

    if predict_only:
        print("unimplemented")
        return
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        train(train_loader, model, opt, nn.MSELoss(), scaler)
        if epoch+1 % 10 == 0:
            torch.save(model.state_dict(), f"model_{epoch}.pth")
        torch.save(model.state_dict(), "model_current.pth")
        
    run.log_model("model_current.pth", name=RUN_NAME)

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

if __name__ == "__main__":
    main(predict_only=False)