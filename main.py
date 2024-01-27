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
    num_epochs = 0
    num_pre_epochs = 0
    image_size = (100, 100)
    batch_size = 200
    save_folder = 'saved_models/cnn'

    # Load data
    train_dataloader, test_dataloader = load_data(('dataset.parquet', 'captchas.parquet'), batch_size=batch_size, image_size=image_size)

    # Load model
    model, max_saved_epoch = load_model(num_classes, device, load_latest=True, save_folder=save_folder)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, max_saved_epoch, train_dataloader, test_dataloader, criterion, optimizer, num_pre_epochs, num_epochs, save_folder, device)

    # Visualize predictions
    save_folder = "saved_models/cnn"  # Adjust the path if necessary
    captcha_visualizer = CaptchaVisualizer(device, num_classes, image_size, save_folder)
    latest_model = captcha_visualizer.load_latest_model()
    captcha_visualizer.visualize_predictions(latest_model, test_dataloader)
