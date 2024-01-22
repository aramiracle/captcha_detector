import torch
import jiwer
from tqdm import tqdm
from utils import load_model, load_data

def predict_captcha(model, images, device, char_set):
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    predicted_classes = torch.argmax(outputs, dim=1)

    # Convert the generator expression to a list comprehension
    predicted_words = [''.join([char_set[class_idx] for class_idx in pred]) for pred in predicted_classes]
    
    # Remove dot ('.') from each predicted word
    predicted_words = [word.replace('.', '') for word in predicted_words]

    return predicted_words

def calculate_accuracy(predictions, ground_truth_labels):
    correct_predictions = sum(1 for pred, truth in zip(predictions, ground_truth_labels) if pred == truth)
    total_samples = len(ground_truth_labels)
    accuracy = correct_predictions / total_samples
    return accuracy

# Function to convert one-hot encoded tensor to alphanumeric text
def onehot_to_alphanumeric(tensor, char_set):
    max_indices = torch.argmax(tensor, dim=1)
    alphanumeric_str = ''.join([char_set[idx.item()] for idx in max_indices.flatten()])
    # Remove dot ('.') from the alphanumeric string
    alphanumeric_str = alphanumeric_str.replace('.', '')
    return alphanumeric_str

def calculate_cer(predictions, ground_truth_labels):
    # Flatten the nested lists
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_ground_truth_labels = [item for sublist in ground_truth_labels for item in sublist]

    # Calculate CER using jiwer.wer
    cer = jiwer.cer(flat_ground_truth_labels, flat_predictions)
    
    return cer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 36  # 26 letters + 10 digits
    image_size = (100, 100)
    char_set = "abcdefghijklmnopqrstuvwxyz0123456789."
    model = load_model(num_classes, device, load_latest=True, save_folder='saved_models/cnn')

    # Load test dataset using load_data function
    _, test_dataloader = load_data(('dataset.csv', 'captchas.csv'), test_size=0.1, random_state=42, batch_size=1000, image_size=image_size)

    # Get predictions for the test dataset
    predictions = []
    ground_truth_labels = []
    with torch.no_grad(), tqdm(total=len(test_dataloader), desc="Testing", unit="batch") as pbar:
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            batch_predictions_text = predict_captcha(model, images, device, char_set)[0]

            predictions.append(batch_predictions_text)
            
            # Convert labels to text before adding to the ground_truth_labels list
            labels_text = [onehot_to_alphanumeric(label, char_set) for label in labels]
            ground_truth_labels.append(labels_text)
            
            pbar.update(1)

    print(f'Predictions: {predictions[0]}')
    print(f'Groun Truths: {ground_truth_labels[0]}')
    # Calculate accuracy and CER
    accuracy = calculate_accuracy(predictions, ground_truth_labels)
    # cer = calculate_cer(predictions, ground_truth_labels)

    print(f"Accuracy: {accuracy * 100:.4f}%")
    # print(f"Character Error Rate (CER): {cer * 100:.4f}%")
