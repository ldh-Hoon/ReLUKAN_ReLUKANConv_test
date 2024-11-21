import torch
import torch.optim as optim

from tqdm import tqdm

from train import *

def grow_train(model, model_name, max_count, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dict):
    early_stopping = EarlyStopping()
    count = 0
    lr = 0.0001
    for epoch in range(num_epochs):
        if epoch != 0:
            if val_accuracy > 0.97:
                if model.expand_count < max_count:
                    model.check_and_expand()
                    model.to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    early_stopping = EarlyStopping()
                    print(f"layer added, {model.expand_count}")

        model.train()
        train_loss = 0.0

        for videos, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")


        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        if early_stopping(val_loss):
            break
        if val_loss >= early_stopping.best_loss: 
            torch.save(model.state_dict(), f'{save_dict}/{model_name}.pth')

        count = model.expand_count
    print("Training complete.")

    return count