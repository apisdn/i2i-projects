import torch
import pickle as pkl
from PIL import Image
from model import Unet
import os
import wandb
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            filename = filename.split("_")[0]

            Image.fromarray(pred).save(os.path.join("predictions", filename + "_fake.png"))
            wandb.log({"Validation Predictions": wandb.Image(os.path.join("predictions", filename + "_fake.png"))})


def main(modelpth, data, foldermode):
    RUN_NAME = "test1_infer"

    run = wandb.init(project="unet-translation-test",
                     job_type="test",
                     notes="Testing a UNet model for image-to-image translation")

    if foldermode:
        print("unimplemented")
        return
    else:
        val_loader = pkl.load(open(data, "rb"))

        # remove these two lines after current testing, bug that caused them to be needed is fixed
        #dataset = val_loader.dataset
        #val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Unet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(modelpth))

    os.makedirs("predictions", exist_ok=True)

    infer(val_loader, model)

if __name__ == "__main__":
    modelpath = "model_99.pth"
    loaderpath = "val_loader.pkl"

    main(modelpath, loaderpath, False)