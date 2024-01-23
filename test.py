import torch
from PIL import Image
from torchvision import transforms
from utils import SquarePadAndResize, load_model
import os
import matplotlib.pyplot as plt

# Function to preprocess the image before feeding it to the model
def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        SquarePadAndResize(image_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to predict captcha using the loaded model
def predict_captcha(model, image_path, image_size, device, char_set):
    model.eval()
    image = preprocess_image(image_path, image_size)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    predicted_classes = torch.argmax(output, 2)
    confidence_per_char = output.max(dim=2).values
    predicted_word = ''.join([char_set[class_idx] for class_idx in predicted_classes.squeeze().tolist()])
    predicted_word = predicted_word.replace('.', '')
    total_confidence = torch.prod(confidence_per_char).item()

    return predicted_word, confidence_per_char.tolist(), total_confidence

def show_all_images_with_predictions(directory_path, model, image_size, device, char_set):
    file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    num_images = len(file_list)
    num_cols = 5  # You can adjust the number of columns in the grid
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle("Captcha Predictions", fontsize=16)

    for i, captcha_image_file in enumerate(file_list):
        captcha_image_path = os.path.join(directory_path, captcha_image_file)

        predicted_word, _, total_confidence = predict_captcha(model, captcha_image_path, image_size, device, char_set)

        ax = axes[i // num_cols, i % num_cols]
        image = Image.open(captcha_image_path)
        ax.imshow(image)
        ax.set_title(f"Predicted: {predicted_word}\nConfidence: {total_confidence:.4f}")
        ax.axis("off")

    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 36  # 26 letters + 10 digits
    image_size = (100, 100)
    char_set = "abcdefghijklmnopqrstuvwxyz0123456789."
    model = load_model(num_classes, device, load_latest=True, save_folder='saved_models/cnn')

    directory_path = 'captcha_examples'

    # Show all images in the directory in a single plot
    show_all_images_with_predictions(directory_path, model, image_size, device, char_set)
