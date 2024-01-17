import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import CNNModel  # Import your CRNN model class
from data_loader import CaptchaDataset  # Import your custom dataset class

def load_data(csv_file, test_size=0.1, random_state=42, batch_size=200, image_size=(100, 100)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    df = pd.read_csv(csv_file)

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    captcha_train_dataset = CaptchaDataset(df_train, transform=transform)
    captcha_test_dataset = CaptchaDataset(df_test, transform=transform)

    train_dataloader = DataLoader(captcha_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(captcha_test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def load_model(num_classes, device, load_latest=True, save_folder="saved_models"):
    model = CNNModel(num_classes).to(device)

    if load_latest:
        # Get a list of all the saved model files
        model_files = [f for f in os.listdir(save_folder) if f.endswith(".pth")]

        if model_files:
            # Find the most recent model file
            latest_model_file = max(model_files, key=lambda x: os.path.getctime(os.path.join(save_folder, x)))

            # Load the parameters of the most recent model
            model_path = os.path.join(save_folder, latest_model_file)
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded the latest model from: {model_path}")
        else:
            print("No saved models found. Training a new model.")

    return model

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, device):
    best_train_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
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
                tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)} - Train Loss: {average_train_loss}")

                if average_train_loss < best_train_loss:
                    best_train_loss = average_train_loss
                    print(f"Epoch {epoch+1} - Best training loss so far: {best_train_loss}")

            save_folder = "saved_models"
            save_path = os.path.join(save_folder, f"captcha_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

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

        tqdm.write(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        tqdm.write(f"  Train Loss: {average_train_loss}")
        tqdm.write(f"  Validation Loss: {average_test_loss}")