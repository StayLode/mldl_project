import torch
from torch import nn, optim
from data.dataloader import prepare_tiny_imagenet
from models.custom_net import CustomNet
from utils.train_utils import train_one_epoch, validate
from utils.visualization import plot_curves

def main():
    # Data
    train_loader, val_loader = prepare_tiny_imagenet()

    # Model, loss, optimizer
    if torch.cuda.is_available():
        model = CustomNet().cuda()
    else:
        model = CustomNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 10
    best_acc = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        best_acc = max(best_acc, val_acc)
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pth')

    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    plot_curves(train_losses, val_losses, train_accs, val_accs)

if __name__ == "__main__":
    main()