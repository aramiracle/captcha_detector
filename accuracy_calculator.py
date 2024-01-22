import torch
import jiwer
from tqdm import tqdm
from utils import load_model, load_data

def predict_captcha(model, images, image_size, device, char_set):
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    predicted_classes = torch.argmax(outputs, 2)
    
    # Extract confidence for the predicted classes
    confidence_per_char = outputs.max(dim=2).values

    predicted_words = [''.join([char_set[class_idx] for class_idx in pred]) for pred in predicted_classes]
    
    total_confidences = [torch.prod(conf).item() for conf in confidence_per_char]

    return predicted_words, confidence_per_char.tolist(), total_confidences


def calculate_accuracy(predictions, ground_truth_labels):
    correct_predictions = sum(1 for pred, truth in zip(predictions, ground_truth_labels) if pred == truth)
    total_samples = len(ground_truth_labels)
    accuracy = correct_predictions / total_samples
    return accuracy

def calculate_cer(predictions, ground_truth_labels):
    total_cer = 0.0
    for pred, truth in zip(predictions, ground_truth_labels):
        cer = jiwer.wer(truth, pred)
        total_cer += cer
    average_cer = total_cer / len(ground_truth_labels)
    return average_cer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 36  # 26 letters + 10 digits
    image_size = (100, 100)
    char_set = "abcdefghijklmnopqrstuvwxyz0123456789."
    model = load_model(num_classes, device, load_latest=True, save_folder='saved_models/cnn')

    # Load test dataset using load_data function
    _, test_dataloader = load_data(('dataset.csv', 'captchas.csv'), test_size=0.1, random_state=42, batch_size=5, image_size=image_size)

    # Get predictions for the test dataset
    predictions = []
    ground_truth_labels = []
    with torch.no_grad(), tqdm(total=len(test_dataloader), desc="Testing", unit="batch") as pbar:
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            batch_predictions = predict_captcha(model, images, image_size, device, char_set)[0]
            predictions.extend(batch_predictions)
            ground_truth_labels.extend(labels)
            pbar.update(1)

    # Calculate accuracy and CER
    accuracy = calculate_accuracy(predictions, ground_truth_labels)
    cer = calculate_cer(predictions, ground_truth_labels)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Character Error Rate (CER): {cer * 100:.2f}%")
