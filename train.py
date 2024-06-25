import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ImageDataset
from model import Unet
from torch.utils.data import DataLoader
from tqdm import tqdm

losses = []
val_losses = []
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(img_dataloader, model, opt, loss_fn, scaler):
    
    loop = tqdm(img_dataloader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        '''
        if len(y.shape) < 4:
            y = y.unsqueeze(1)  # add channel dimension

        if len(x.shape) < 4:
            x = x.unsqueeze(1)  # add channel dimension
        '''

        with torch.cuda.amp.autocast():
            preds = model(x)
            loss = loss_fn(preds, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())


def main(predict_only=False):
    # parameters
    in_chan = 3
    out_chan = 3
    learning_rate = 1e-4
    batch_size = 1
    num_epochs = 10
    loss_fn = nn.CrossEntropyLoss()

    model = Unet(in_channels=in_chan, out_channels=out_chan).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

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
        torch.save(model.state_dict(), f"model_{epoch}.pth")
        print("Model saved")

        print("Validation")
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                preds = model(x)
                loss = loss_fn(preds, y)
                val_losses.append(loss.item())

    print("Training complete")
    print("Mean validation loss: ", sum(val_losses) / len(val_losses))

if __name__ == "__main__":
    main(predict_only=False)