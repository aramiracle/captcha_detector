import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data, load_model, train_model
from visualize import CaptchaVisualizer

if __name__ == "__main__":
    # Set device and hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 36  # 26 letters + 10 digits
    learning_rate = 0.001
    num_epochs = 10
    image_size = (100, 100)
    batch_size = 200

    # Load data
    train_dataloader, test_dataloader = load_data('dataset.csv', batch_size=batch_size, image_size=image_size)

    # Load model
    model = load_model(num_classes, device, load_latest=True, save_folder='saved_models/cnn')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs, device)

    # Visualize predictions
    save_folder = "saved_models"  # Adjust the path if necessary
    captcha_visualizer = CaptchaVisualizer(device, num_classes, image_size, save_folder)
    latest_model = captcha_visualizer.load_latest_model()
    test_dataloader = captcha_visualizer.load_test_data('dataset.csv')
    captcha_visualizer.visualize_predictions(latest_model, test_dataloader)
