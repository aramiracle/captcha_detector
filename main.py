import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import CRNNModel  # Import your CRNN model class
from data_loader import CaptchaDataset  # Import your custom dataset class

# Set device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 36  # 26 letters + 10 digits
max_length = 10
image_size = (100, 100)
batch_size = 200

learning_rate = 0.001
num_epochs = 10

# Create dataset and split into training and testing sets
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

os.makedirs('saved_models', exist_ok=True)

df = pd.read_csv('dataset.csv')

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets and dataloaders for training and testing
captcha_train_dataset = CaptchaDataset(df_train, transform=transform)
captcha_test_dataset = CaptchaDataset(df_test, transform=transform)

train_dataloader = DataLoader(captcha_train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(captcha_test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle the test set

# Create the CRNN model, CTC loss function, and optimizer
model = CRNNModel(num_classes).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_train_loss = float('inf')  # Initialize best train loss to positive infinity

# Training loop with tqdm progress bars
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
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
                
            # Check if the current training loss is better than the best so far
        if average_train_loss < best_train_loss:
            best_train_loss = average_train_loss
            print(f"Epoch {epoch+1} - Best training loss so far: {best_train_loss}")

        # Save the trained model after each epoch
        save_folder = "saved_models"
        save_path = os.path.join(save_folder, f"captcha_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
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

    # Print the summary at the end of each epoch
    tqdm.write(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
    tqdm.write(f"  Train Loss: {average_train_loss}")
    tqdm.write(f"  Validation Loss: {average_test_loss}")
