# import matplotlib.pyplot as plt
import json
import os
import torch
from torchmetrics.classification import BinaryJaccardIndex
import torch.optim as optim
from torch.utils.data import DataLoader
from u_net import UNet
from bce_diceloss import DiceBCELoss


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def get_avg_jaccard(model, val_loader, binary_jaccard) -> float:
    with torch.no_grad():
        num = 0.0
        for inputs, labels in val_loader:
            num += binary_jaccard(model(inputs).round(), labels)
        return num / len(val_loader)


def get_avg_loss(model, val_loader, loss_fn):
    with torch.no_grad():
        avg_loss = 0.0
        for inputs, labels in val_loader:
            loss = loss_fn(model(inputs), labels).item()
            avg_loss += loss
        return avg_loss / len(val_loader)


def training_loop(model, optimizer, scheduler, train_loader, val_loader, epochs):
    loss_fn = DiceBCELoss().cuda()
    metric = BinaryJaccardIndex().cuda()
    save_path = os.path.join("./out/", params["name"])
    best_loss = 1e9
    for epoch in range(1, epochs + 1):
        avg_loss_train = 0.0
        current = 1
        total = len(training_loader)
        for inputs, labels in train_loader:
            printProgressBar(
                current,
                total,
                prefix=f"Epoch: {
                    epoch}",
                suffix=f"{current}/{total}",
            )
            loss = loss_fn(model(inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss_train += loss.item()
            current += 1
        val_loss = get_avg_loss(model, val_loader, loss_fn)
        if val_loss < best_loss:
            print(f"New best validation loss! Average DiceBCELoss: {val_loss}")
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
        scheduler.step()
        print(f"""\nEpoch: {epoch}
        Avg train loss: {avg_loss_train/total}
        Avg val loss: {val_loss},
        Avg val jaccard index: {get_avg_jaccard(model, val_loader, metric)}
        """)


if __name__ == "__main__":
    params = json.load(open("./params.json"))
    batch_size = params["batch_size"]
    training_dataset = torch.load("./train_data.pt")
    val_dataset = torch.load("./test_data.pt")
    training_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    model = UNet(out_ch=params["filters"]).cuda()
    optimizer = optim.SGD(
        model.parameters(), lr=params["learning_rate"], weight_decay=1e-5, momentum=0.95
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)
    training_loop(
        model, optimizer, scheduler, training_loader, val_loader, params["epochs"]
    )
