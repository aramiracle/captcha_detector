import torch
from PIL import Image
from torchvision import transforms
from utils import load_model

# Function to preprocess the image before feeding it to the model
def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Convert the image to grayscale
        transforms.ToTensor(),
        # Add any other necessary transformations based on your model's input requirements
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
    
    # Extract confidence for the predicted classes
    confidence_per_char = output.max(dim=2).values

    predicted_word = ''.join([char_set[class_idx] for class_idx in predicted_classes.squeeze().tolist()])
    
    total_confidence = torch.prod(confidence_per_char).item()

    return predicted_word, confidence_per_char.tolist(), total_confidence

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 36  # 26 letters + 10 digits
    image_size = (100, 100)
    char_set = "abcdefghijklmnopqrstuvwxyz0123456789."
    model = load_model(num_classes, device, load_latest=True, save_folder='saved_models/cnn')

    captcha_image_path = 'test_examples/example03.png'

    predicted_word, confidence_per_char, total_confidence = predict_captcha(model, captcha_image_path, image_size, device, char_set)

    print(f"Predicted word: {predicted_word}")
    print(f"Confidence per character: {confidence_per_char}")
    print(f"Total confidence: {total_confidence}")