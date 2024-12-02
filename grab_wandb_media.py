import wandb
import os

api = wandb.Api()
run1 = api.run("apisdn/unet-translation-3to1/84roc6bw") # 3 to 1
run2 = api.run("apisdn/unet-translation-3to1/5fhn71wo") # 1 to 1

os.mkdir('unet3to1')
os.chdir('unet3to1')

for file in run1.files():
    file.download()

os.chdir('..')
os.mkdir('unet1to1')
os.chdir('unet1to1')

for file in run2.files():
    file.download()