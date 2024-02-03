import gradio as gr
from PIL import Image
from utils import SquarePadAndResize, load_model
import torchvision.transforms as transforms
import torch
import os

# Define a global variable to track whether the model has been loaded
model_loaded = False

# Global variable to store the loaded model
captcha_model = None

# Function to preprocess the image before feeding it to the model
def preprocess_image(image):
    image_size = (100, 100)
    transform = transforms.Compose([
        SquarePadAndResize(image_size),
        transforms.ToTensor()
    ])
    image = Image.fromarray(image.astype('uint8'))
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to predict captcha using the loaded model
def predict_captcha(model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_set = "abcdefghijklmnopqrstuvwxyz0123456789."
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    predicted_classes = torch.argmax(output, 2)
    
    # Extract confidence for the predicted classes
    confidence_per_char = output.max(dim=2).values

    predicted_text = ''.join([char_set[class_idx] for class_idx in predicted_classes.squeeze().tolist()])
    
    total_confidence = torch.prod(confidence_per_char).item()

    return predicted_text, total_confidence

# Function to perform captcha detection using the model
def detect_captcha(image):
    global model_loaded, captcha_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_data = preprocess_image(image)

    # Load the model only if it hasn't been loaded yet
    if not model_loaded:
        num_classes = 36  # 26 letters + 10 digits
        save_folder = 'saved_models/cnn'
        captcha_model, _ = load_model(num_classes, device, load_latest=True, save_folder=save_folder)
        model_loaded = True

    # Perform prediction using your captcha model
    predicted_text, total_confidence = predict_captcha(captcha_model, input_data)

    # Remove dots from the predicted text
    predicted_text_without_dots = predicted_text.replace('.', '')

    return predicted_text_without_dots, {'Total confidence': total_confidence}

iface = gr.Interface(
    fn=detect_captcha,
    inputs=gr.Image(),
    outputs=[
        gr.Textbox(label="Output 🤖"),
        gr.Label(label="Total Confidence 🔍", num_top_classes=1)
    ],
    examples=[
        os.path.join('captcha_examples', 'example01.png'),
        os.path.join('captcha_examples', 'example02.png'),
        os.path.join('captcha_examples', 'example03.png'),
        os.path.join('captcha_examples', 'example04.png'),
        os.path.join('captcha_examples', 'example05.png'),
    ],
    title="Captcha Detection 🕵️‍♂️",
    description="Welcome to the Captcha Detection System! 🚀\n\nUpload an image with a captcha, and let our model decipher the text. The model is trained to recognize letters and digits in the captcha. 🧠💡\n\nSee how accurate it is and enjoy the magic of AI! ✨🔮",
)

# Launch the interface
iface.launch()
