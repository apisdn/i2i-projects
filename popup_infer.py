import torch
import pickle as pkl
from PIL import Image
from model import Unet
import os
import wandb
from torch.utils.data import DataLoader

VAL_LOADER_1 = "unet3to1/val_loader.pkl"
VAL_LOADER_2 = "unet1to1/val_loader.pkl"

SAVE_DIR_1 = "unet3to1/predictions"
SAVE_DIR_2 = "unet1to1/predictions"
os.makedirs(SAVE_DIR_1, exist_ok=True)
os.makedirs(SAVE_DIR_2, exist_ok=True)

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
            filename = filename.split(".")[0]

            Image.fromarray(pred).save(os.path.join(SAVE_DIR_2, filename + "_fake.png"))
            #wandb.log({"Validation Predictions": wandb.Image(os.path.join("predictions", filename + "_fake.png"))})


def main(foldermode):
    RUN_NAME = "unet-validate-3v1"

    run = wandb.init(project="unet-translation-3to1",
                     job_type="test",
                     notes="Test using previous models due to file corruption",
                     entity="apisdn",
                     name=RUN_NAME)

    if foldermode:
        print("unimplemented")
        return
    else:
        #val_loader = pkl.load(open(VAL_LOADER_1, "rb"))
        val_loader = pkl.load(open(VAL_LOADER_2, "rb"))

        # remove these two lines after current testing, bug that caused them to be needed is fixed
        #dataset = val_loader.dataset
        #val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    #modelpath = "apisdn/unet-translation-3to1/experiment3to1:v1"
    modelpath = "apisdn/unet-translation-3to1/experiment3to1:v2"

    downloaded_model_path = run.use_model(name=modelpath)
    print(downloaded_model_path)

    model = Unet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(downloaded_model_path, map_location=torch.device('cpu')))

    os.makedirs(SAVE_DIR_1, exist_ok=True)

    infer(val_loader, model)

if __name__ == "__main__":
    main(False)