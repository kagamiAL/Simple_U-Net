import matplotlib.pyplot as plt
import torch
import json
from torchvision import transforms
from u_net import UNet
from torch.utils.data import DataLoader

amt = 3
params = json.load(open("./params.json"))
dataset = torch.load("./test_data.pt")
dataloader = DataLoader(dataset, shuffle=True, batch_size=amt)
model = UNet(out_ch=params["filters"]).cuda()
model.load_state_dict(torch.load("./out/cloud_model_3"))

mean = [151.1672, 141.4257, 139.8164]
std = [55.9709, 52.2902, 55.2580]
inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
)


with torch.no_grad():
    rows = amt
    columns = 3
    iterate = iter(dataloader)
    inputs, labels = next(iterate)
    while True and inputs is not None:
        predicteds = model(inputs)
        fig = plt.figure(figsize=(10, 10))
        for i in range(rows):
            image = inv_normalize(inputs[i]).div(255).permute(1, 2, 0).cpu()
            label = labels[i][0].cpu()
            predicted = predicteds[i][0].round().cpu()

            fig.add_subplot(rows, columns, i * rows + 1)
            plt.imshow(image)
            plt.title("Input")

            fig.add_subplot(rows, columns, i * rows + 2)
            plt.imshow(label)
            plt.title("Label")

            fig.add_subplot(rows, columns, i * rows + 3)
            plt.imshow(predicted)
            plt.title("Predicted")

        plt.show()
        inputs, labels = next(iterate)
