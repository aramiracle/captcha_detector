import gradio as gr
from PIL import Image
from utils import load_model
import torchvision.transforms as transforms
import torch

# Function to preprocess the image before feeding it to the model
def preprocess_image(image):
    image_size = (100, 100)
    transform = transforms.Compose([
        transforms.Resize(image_size),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_data = preprocess_image(image)

    num_classes = 36  # 26 letters + 10 digits
    save_folder = 'saved_models/cnn'

    captcha_model = load_model(num_classes, device, load_latest=True, save_folder=save_folder)

    # Perform prediction using your captcha model
    predicted_text, total_confidence = predict_captcha(captcha_model, input_data)

    # Remove dots from the predicted text
    predicted_text_without_dots = predicted_text.replace('.', '')

    return predicted_text_without_dots, {'Total confidence': total_confidence}

# Create a Gradio interface
iface = gr.Interface(
    fn=detect_captcha,
    inputs=gr.Image(),
    outputs=[
        gr.Textbox(label="Predicted Text"),
        gr.Label(label="Total Confidence", num_top_classes=1)
    ],
    title="Captcha Detection",
    description="Upload an image with a captcha to detect the text.",
)

# Launch the interface
iface.launch()
