import torch
import pickle as pkl
from PIL import Image
from model import Unet
import os

#predictions = []
def infer(img_dataloader, model):
    model.eval()
    with torch.no_grad():
        for x,y,imgidx in img_dataloader:
            x = x.to("cpu")

            pred = model(x)

            datas = img_dataloader.dataset
            filename = datas.dataset.get_filenames(imgidx)

            pred = pred.squeeze(0).permute(1, 2, 0).numpy()
            pred = (pred * 255).astype("uint8")

            filename = os.path.basename(filename)
            filename = filename.split("_")[0]

            Image.fromarray(pred).save(os.path.join("predictions", filename + "_fake.png"))
        
    # get the filenames of the images from the dataset in the dataloader and save the predictions as pngs with the same filenames
    '''
    for idx, (x, y) in enumerate(img_dataloader):
        filename = img_dataloader.dataset.get_filenames(x)
        pred = predictions[idx]
        pred = pred.squeeze(0).permute(1, 2, 0).numpy()
        pred = (pred * 255).astype("uint8")
        Image.fromarray(pred).save(filename)
    '''

def main(modelpth, data, foldermode):
    if foldermode:
        print("unimplemented")
        return
    else:
        val_loader = pkl.load(open(data, "rb"))

    model = Unet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(modelpth))

    os.makedirs("predictions", exist_ok=True)

    infer(val_loader, model)

if __name__ == "__main__":
    modelpath = "model_9.pth"
    loaderpath = "val_loader.pkl"

    main(modelpath, loaderpath, False)