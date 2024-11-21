import os, importlib, json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split

from train import *
from grow_train import *
from model import *


class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    transform = transforms.Compose([
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = MNISTDataset(root='./data', train=True, transform=transform)
    test_dataset = MNISTDataset(root='./data', train=False, transform=transform)


    train_size = int(0.9 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    print(train_size, val_size, len(test_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    num_epochs = 10
    BATCH_SIZE = 16
    learning_rate = 0.0001

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    models = [
        "CNN_MLP",
        "CNN_ReLUKAN_grow",
        "CNN_ReLUKAN_nogrow",
        "CNN_MLP_grow",
        "CNN_MLP_nogrow",

        "CNN_ReLUKAN",
        "CNN_SiLUKAN",
        "DenseMLP",
        "DenseReLUKAN",
        "ReLUKAN_Conv_MLP",
        "ReLUKAN_Conv_ReLUKAN",
        "ViT"
    ]

    save_dict = "./trained"
    criterion = nn.CrossEntropyLoss()
    test_log = dict()

    for model_name in models:
        print(model_name)
        if "grow" in model_name:
            max_count = 3
            if "no" in model_name:
                module = importlib.import_module(f"model.{model_name}")
                model_class = getattr(module, model_name)
                model = model_class(max_count).to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                train(model, model_name, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dict)
            else:
                module = importlib.import_module(f"model.{model_name}")
                model_class = getattr(module, model_name)
                model = model_class().to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                grow_train(model, model_name, max_count, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dict)
        else:
            module = importlib.import_module(f"model.{model_name}")
            model_class = getattr(module, model_name)
            model = model_class().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train(model, model_name, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dict)
        
        acc = test(model, test_loader, criterion, device)
        test_log[model_name] = acc
    
    print(test_log)

    with open('MNIST_test_log.json', 'w') as json_file:
        json.dump(test_log, json_file, indent=4)