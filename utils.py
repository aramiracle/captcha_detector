import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import CNNModel  # Import your CRNN model class
from data_loader import CaptchaDataset  # Import your custom dataset class

class SquarePadAndResize(object):
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        # Pad the image to make it square
        w, h = img.size
        max_size = max(w, h)
        new_img = Image.new('RGB', (max_size, max_size), color=self.fill)
        new_img.paste(img, ((max_size - w) // 2, (max_size - h) // 2))

        # Resize the image to the desired size
        new_img = new_img.resize(self.size, Image.BICUBIC)

        return new_img

def load_data(csv_files, test_size=0.1, random_state=42, batch_size=200, image_size=(100, 100)):
    transform = transforms.Compose([
        SquarePadAndResize(image_size),
        transforms.ToTensor()
    ])

    df_downloaded = pd.read_csv(csv_files[0])
    df_generated = pd.read_csv(csv_files[1])
    df = pd.concat([df_downloaded, df_generated])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    
    captcha_train_dataset = CaptchaDataset(df_train, transform=transform)
    captcha_test_dataset = CaptchaDataset(df_test, transform=transform)

    train_dataloader = DataLoader(captcha_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(captcha_test_dataset, batch_size=5, shuffle=False)

    return train_dataloader, test_dataloader


def load_model(num_classes, device, load_latest=True, save_folder="saved_models"):
    model = CNNModel(num_classes).to(device)

    if load_latest:
        # Get a list of all the saved model files
        model_files = [f for f in os.listdir(save_folder) if f.startswith("captcha_model_epoch_")]

        if model_files:
            # Extract epoch numbers from the file names
            epochs = [int(file.split("_")[-1].split(".")[0]) for file in model_files]

            # Find the maximum epoch number
            max_epoch = max(epochs)

            # Load the parameters of the model with the maximum epoch number
            latest_model_file = f"captcha_model_epoch_{max_epoch}.pth"
            model_path = os.path.join(save_folder, latest_model_file)
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded the latest model from: {model_path}")
        else:
            max_epoch = 0
            print("No saved models found. Training a new model.")

    return model, max_epoch

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def train_model(model, start_epoch, train_dataloader, test_dataloader, criterion, optimizer, pretrain_epochs, train_epochs, save_folder, device):
    best_train_loss = float('inf')

    # Pretraining phase
    model.train()
    set_requires_grad(model.cnn, False)  # Freeze CNN layers

    for epoch in range(pretrain_epochs):
        pretrain_loss = 0

        with tqdm(total=len(train_dataloader), desc=f"Pretrain Epoch {epoch + 1}/{pretrain_epochs}", unit="batch") as pbar:
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(labels.to(torch.float32), outputs)

                loss.backward()
                optimizer.step()

                pretrain_loss += loss.item()
                pbar.update(1)

                average_pretrain_loss = pretrain_loss / (batch_idx + 1)
                pbar.set_postfix({"Pretrain Loss": average_pretrain_loss})
                tqdm.write(f"Pretrain Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)} - Pretrain Loss: {average_pretrain_loss}")

    set_requires_grad(model.cnn, True)  # Unfreeze CNN layers

    # Training phase
    for epoch in range(pretrain_epochs + start_epoch, pretrain_epochs + start_epoch + train_epochs):
        model.train()
        train_loss = 0

        with tqdm(total=len(train_dataloader), desc=f"Train Epoch {epoch + 1}/{pretrain_epochs + start_epoch + train_epochs}", unit="batch") as pbar:
            for batch_idx, (images, labels) in enumerate(train_dataloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(labels.to(torch.float32), outputs)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.update(1)

                average_train_loss = train_loss / (batch_idx + 1)
                pbar.set_postfix({"Train Loss": average_train_loss})
                tqdm.write(f"Train Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)} - Train Loss: {average_train_loss}")

                if average_train_loss < best_train_loss:
                    best_train_loss = average_train_loss
                    print(f"Train Epoch {epoch+1} - Best training loss so far: {best_train_loss}")

            save_path = os.path.join(save_folder, f"captcha_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

        # Validation phase
        model.eval()
        test_loss = 0

        with torch.no_grad(), tqdm(total=len(test_dataloader), desc="Validation", unit="batch") as pbar:
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(labels.to(torch.float32), outputs)

                test_loss += loss.item()
                pbar.update(1)

            average_test_loss = test_loss / len(test_dataloader)
            pbar.set_postfix({"Train Loss": average_train_loss, "Validation Loss": average_test_loss})

        tqdm.write(f"\nEpoch {epoch+1}/{pretrain_epochs + train_epochs} Summary:")
        tqdm.write(f"  Train Loss: {average_train_loss}")
        tqdm.write(f"  Validation Loss: {average_test_loss}")