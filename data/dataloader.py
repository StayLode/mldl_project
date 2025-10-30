import os
import shutil
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

def prepare_tiny_imagenet(root_dir='tiny-imagenet/tiny-imagenet-200'):
    """Prepare Tiny ImageNet validation folders and return dataloaders."""
    val_annot = os.path.join(root_dir, 'val', 'val_annotations.txt')

    # Create class subfolders for validation
    with open('tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

            shutil.copyfile(f'tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

    shutil.rmtree('tiny-imagenet/tiny-imagenet-200/val/images')
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=os.path.join(root_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(root_dir, 'val'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader