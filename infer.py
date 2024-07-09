import torch
import pickle as pkl
from PIL import Image
from model import Unet
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def main(modelpth, data, foldermode):
    if foldermode:
        print("unimplemented")
        return
    else:
        val_loader = pkl.load(open(data, "rb"))

    model = Unet(in_channels=3, out_channels=3).to(device)
    model.load_state_dict(torch.load(modelpth))

    os.makedirs("predictions", exist_ok=True)

    infer(val_loader, model)

if __name__ == "__main__":
    modelpath = "model_99.pth"
    loaderpath = "val_loader.pkl"

    main(modelpath, loaderpath, False)