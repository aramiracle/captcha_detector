import os
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from model import CNNModel  # Import your CRNN model class
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

                # Convert model outputs to indices
                outputs_indices = torch.argmax(outputs, dim=2)
                labels_indices = torch.argmax(labels, dim=2)

                # Decode true labels
                true_labels = [self.decode_labels(label) for label in labels_indices]

                # Decode predicted labels and get confidence
                predictions = [self.decode_predictions(j, outputs_indices) for j in range(images.size(0))]
                total_confidences = [np.prod(self.get_confidence_per_char(j, outputs, outputs_indices[j]))
                                    for j in range(images.size(0))]

                # Check correctness of predictions
                correct_predictions = [true[0] == pred for true, pred in zip(true_labels, predictions)]

                # Visualize each batch in a single window
                fig, axs = plt.subplots(images.size(0), 1, figsize=(8, 2 * images.size(0)))

                for j in range(images.size(0)):
                    ax = axs[j]
                    image = transforms.ToPILImage()(images[j].cpu())
                    title = f"True: {true_labels[j][0]}, Predicted: {predictions[j]}, " \
                            f"Total confidence: {total_confidences[j]:.4f}, " \
                            f"{'Correct' if correct_predictions[j] else 'Incorrect'}"
                    ax.imshow(np.array(image))
                    ax.set_title(title)

                plt.tight_layout()
                plt.show()
