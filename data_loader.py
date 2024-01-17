import torch
from torch.utils.data import Dataset
import torch.nn.functional as F 
from PIL import Image
import io
import ast
import re

class CaptchaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.char_to_index = self.create_char_to_index_mapping()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_bytes = ast.literal_eval(self.dataframe.iloc[idx]['image'])['bytes']  # Use iloc to access rows by index
        
        label_string = self.dataframe.iloc[idx]['text']  # Use iloc to access rows by index
        label_chars = re.findall(r"'(.*?)'", label_string)[0].lower()
        label = self.convert_label_to_sequence(label_chars)

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # Assuming it's a grayscale image
        if self.transform:
            img = self.transform(img)

        return img, label

    def create_char_to_index_mapping(self):
        characters = "abcdefghijklmnopqrstuvwxyz0123456789"
        return {char: index for index, char in enumerate(characters)}

    def convert_label_to_sequence(self, label_chars, max_length=10):  # Set max_length to the maximum label length
        label_sequence = [self.char_to_index[char] for char in label_chars]

        # Pad the sequence with zeros if its length is less than max_length
        pad_length = max(0, max_length - len(label_sequence))
        label_sequence += [36] * pad_length

        label_sequence = torch.tensor(label_sequence, dtype=torch.long)
        label_sequence = F.one_hot(label_sequence, num_classes=36 + 1).view(max_length ,36 + 1)
        
        return label_sequence