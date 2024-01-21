import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
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
        flag = False
        img_dict = self.dataframe.iloc[idx]['image']
        if isinstance(img_dict, str):
            img_bytes = ast.literal_eval(img_dict)['bytes']
        else:
            flag =True
            img_bytes = img_dict.values[0]['bytes']

        label_string = self.dataframe.iloc[idx]['text']
        label_chars = re.findall(r"'(.*?)'", label_string)[0].lower()
        label = self.convert_label_to_sequence(label_chars)

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = transforms.ToTensor()(img)

        if torch.all(img_tensor == 0):
            print("Image is black")
            if flag:
                print("Generated captchas loaded but it is black")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        return img, label

    def create_char_to_index_mapping(self):
        characters = "abcdefghijklmnopqrstuvwxyz0123456789"
        return {char: index for index, char in enumerate(characters)}

    def convert_label_to_sequence(self, label_chars, max_length=10):
        label_sequence = [self.char_to_index[char] for char in label_chars]
        pad_length = max(0, max_length - len(label_sequence))
        label_sequence += [36] * pad_length
        label_sequence = torch.tensor(label_sequence, dtype=torch.long)
        label_sequence = F.one_hot(label_sequence, num_classes=36 + 1).view(max_length, 36 + 1)
        
        return label_sequence
