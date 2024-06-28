import torch
import pickle as pkl
from PIL import Image

predictions = []
def infer(img_dataloader, model):
    model.eval()
    with torch.no_grad():
        for x, y in img_dataloader:
            x = x.to("cpu")

            pred = model(x)
            predictions.append(pred)
        
    # get the filenames of the images from the dataset in the dataloader and save the predictions as pngs with the same filenames
    for idx, (x, y) in enumerate(img_dataloader):
        filename = img_dataloader.dataset.get_filenames(x)
        pred = predictions[idx]
        pred = pred.squeeze(0).permute(1, 2, 0).numpy()
        pred = (pred * 255).astype("uint8")
        Image.fromarray(pred).save(filename)


def main(model, data, foldermode):
    if foldermode:
        print("unimplemented")
        return
    else:
        val_loader = pkl.load(open(data, "rb"))

    model.load_state_dict(torch.load(model))

    infer(val_loader, model)

if __name__ == "__main__":
    modelpath = "model_9.pth"
    loaderpath = "val_loader.pkl"

    main()