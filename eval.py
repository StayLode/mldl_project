import torch
from torch import nn
from data.dataloader import prepare_tiny_imagenet
from models.custom_net import CustomNet
from utils.train_utils import validate

def main():
    # Data
    _, val_loader = prepare_tiny_imagenet()

    # Model
    model = CustomNet().cuda()
    model.load_state_dict(torch.load('checkpoints/model_epoch_10.pth'))

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    validate(model, val_loader, criterion)

if __name__ == "__main__":
    main()