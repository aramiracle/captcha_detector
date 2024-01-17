import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from model import CNNModel  # Import your CRNN model class
from data_loader import CaptchaDataset  # Import your custom dataset class
import matplotlib.pyplot as plt
import numpy as np

class CaptchaVisualizer:
    def __init__(self, device, num_classes, image_size, save_folder):
        self.device = device
        self.num_classes = num_classes
        self.image_size = image_size
        self.save_folder = save_folder
        self.index_to_char_mapping = self.create_index_to_char_mapping()

    def create_index_to_char_mapping(self):
        characters = "abcdefghijklmnopqrstuvwxyz0123456789."
        return {index: char for index, char in enumerate(characters)}

    def load_latest_model(self):
        model_files = [f for f in os.listdir(self.save_folder) if f.startswith("captcha_model_epoch_")]
        latest_model_file = max(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        # Load the pre-trained model
        model = CNNModel(self.num_classes).to(self.device)
        model_path = os.path.join(self.save_folder, latest_model_file)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode

        return model

    def load_test_data(self, csv_file, test_size=0.1, random_state=42, batch_size=5):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        df = pd.read_csv(csv_file)
        _, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

        captcha_test_dataset = CaptchaDataset(df_test, transform=transform)
        test_dataloader = DataLoader(captcha_test_dataset, batch_size=batch_size, shuffle=True)

        return test_dataloader

    def decode_labels(self, labels_indices):
        return ["".join([self.index_to_char_mapping[idx.item()] for idx in labels_indices])]

    def decode_predictions(self, i, outputs_indices):
        return "".join([self.index_to_char_mapping[idx.item()] for idx in outputs_indices[i]])

    def get_confidence_per_char(self, i, softmax_probs, pred):
        return [softmax_probs[i, j, pred[j]].item() for j in range(len(pred))]

    def visualize_image(self, image):
        plt.imshow(np.array(image))
        plt.show()

    def visualize_predictions(self, model, test_dataloader):
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass to get predictions
                model.eval()
                outputs = model(images)
                
                # Convert one-hot encoded predictions to indices
                outputs_indices = torch.argmax(outputs, dim=2)
                labels_indices = torch.argmax(labels, dim=2)

                # Get softmax probabilities for confidence
                softmax_probs = torch.nn.functional.softmax(outputs, dim=2)
                
                # Decode true labels
                true_labels = [self.decode_labels(label) for label in labels_indices]

                # Decode predicted labels and get confidence
                predictions = [self.decode_predictions(j, outputs_indices) for j in range(images.size(0))]
                total_confidences = [np.prod(self.get_confidence_per_char(j, softmax_probs, outputs_indices[j]))
                                    for j in range(images.size(0))]

                # Calculate negative logarithm (base 10 or base e) of the confidence values
                neg_log_confidences = [-np.log(confidence) for confidence in total_confidences]

                # Visualize each batch in a single window
                fig, axs = plt.subplots(images.size(0), 1, figsize=(8, 2 * images.size(0)))

                for j in range(images.size(0)):
                    ax = axs[j]
                    image = transforms.ToPILImage()(images[j].cpu())
                    ax.imshow(np.array(image))
                    ax.set_title(f"True: {true_labels[j]}, Predicted: {predictions[j]}, -log(Confidence): {neg_log_confidences[j]:.2f}")

                    # Create a separate text box to display confidence information
                    char_confidences = self.get_confidence_per_char(j, softmax_probs, outputs_indices[j])
                    textstr = "\n".join([f"Char: {char}, Confidence: {confidence:.2f}" for char, confidence in zip(predictions[j], char_confidences)])
                    ax.text(1.05, 0.5, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    # Set device and hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 36  # 26 letters + 10 digits
    image_size = (100, 100)
    save_folder = "saved_models/cnn"

    # Create CaptchaVisualizer instance
    captcha_visualizer = CaptchaVisualizer(device, num_classes, image_size, save_folder)

    # Load the latest model
    model = captcha_visualizer.load_latest_model()

    # Load test data
    test_dataloader = captcha_visualizer.load_test_data('dataset.csv')

    # Visualize predictions
    captcha_visualizer.visualize_predictions(model, test_dataloader)
