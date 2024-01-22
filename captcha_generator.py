import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from math import sin, radians
from torchvision import transforms
import pandas as pd
from io import BytesIO
from tqdm import tqdm
import os

def generate_random_string(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def draw_text(draw, image, text, font, position, color, angle, margin_x=5, margin_y=5):
    # Use a temporary image for size calculation
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Get the bounding box of the text without margins
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Adjust margin based on rotation angle
    adjusted_margin_x = margin_x + int(abs(text_height * sin(radians(angle))))
    adjusted_margin_y = margin_y + int(abs(text_width * sin(radians(angle))))
    
    # Create a temporary image to rotate the text
    rotated_text_img = Image.new('RGBA', (text_width + 2 * adjusted_margin_x, text_height + 2 * adjusted_margin_y), color=(255, 255, 255, 0))
    rotated_text_draw = ImageDraw.Draw(rotated_text_img)
    
    rotated_text_draw.text((adjusted_margin_x, adjusted_margin_y), text, font=font, fill=color)
    
    # Rotate the temporary image
    rotated_text_img = rotated_text_img.rotate(angle, expand=1)
    
    # Get the bounding box for the rotated text
    rotated_bbox = rotated_text_img.getbbox()
    
    # Calculate the adjusted position for the pasted text with margins
    adjusted_position = (position[0] - rotated_bbox[0] + margin_x,
                         position[1] - rotated_bbox[1])
    
    # Paste the rotated text onto the main image using the mask
    image.paste(rotated_text_img, adjusted_position, mask=rotated_text_img)


def draw_spots(draw, width, height, num_spots):
    for _ in range(num_spots):
        spot_position = (random.randint(0, width), random.randint(0, height))
        spot_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.point(spot_position, fill=spot_color)

def draw_lines(draw, width, height, num_lines):
    for _ in range(num_lines):
        line_start = (random.randint(0, width), random.randint(0, height))
        line_end = (random.randint(0, width), random.randint(0, height))
        line_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.line([line_start, line_end], fill=line_color, width=2)

def elastic_transform(image, alpha, sigma):
    random_state = random.getstate()
    random.seed(42)
    
    # Convert the image to a PyTorch tensor
    image_tensor = transforms.ToTensor()(image)
    
    # Apply elastic transform
    elastic_transformer = transforms.ElasticTransform((alpha, sigma))
    transformed_tensor = elastic_transformer(image_tensor)
    
    # Convert the transformed tensor back to a PIL Image
    transformed_image = transforms.ToPILImage()(transformed_tensor)
    
    # Restore the random state
    random.setstate(random_state)
    
    return transformed_image

def generate_captcha():
    # Increase the size of the captcha image
    width, height = 360, 120
    captcha = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(captcha)

    # Set a specific font size and use Carlito font
    font_size = 60
    font_path = "Carlito-Bold.ttf"
    font = ImageFont.truetype(font_path, font_size)

    captcha_text = generate_random_string(random.randint(6, 10))
    
    # Adjust the starting position calculation to place text at the upper part
    text_position = ((width - captcha.width) // 2, 10)  # Adjust the vertical position here
    text_color = (random.randint(0, 64), random.randint(0, 64), random.randint(0, 64))
    text_angle = random.randint(-10, 10)

    # Draw the rotated text with margins
    draw_text(draw, captcha, captcha_text, font, text_position, text_color, text_angle, margin_x=25, margin_y=25)

    draw_spots(draw, width, height, 10000)
    draw_lines(draw, width, height, 20)

    # Convert image to RGBA before applying elastic transformation
    captcha = captcha.convert('RGBA')

    # Apply random elastic transformation
    captcha = elastic_transform(captcha, alpha=30.0, sigma=3.0)

    # Convert image back to RGB after elastic transformation
    captcha = captcha.convert('RGB')

    # Resize to half size to save memory
    captcha = captcha.resize((width // 3 * 2, height // 3 * 2))

    # Apply image smoothing
    captcha = captcha.filter(ImageFilter.SMOOTH)

    return captcha, captcha_text

def generate_captcha_data(num_captchas):
    captcha_data = []

    for i in tqdm(range(num_captchas), desc="Generating Captchas", unit="captcha"):
        # Generate captcha
        captcha, captcha_text = generate_captcha()

        # Convert captcha image to binary bytes
        image_bytes = BytesIO()
        captcha.save(image_bytes, format='PNG')
        image_bytes.seek(0)  # Reset the buffer position to the beginning

        # Append captcha data to the list
        captcha_data.append({'image':{'bytes': image_bytes.getvalue(), 'path': f'{captcha_text}.png'}, 'text': f"This is '{captcha_text}'"})

    return captcha_data

if __name__ == "__main__":
    num_captchas = 30000
    captcha_data = generate_captcha_data(num_captchas)

    # Create DataFrame from captcha data
    df = pd.DataFrame(captcha_data)

    # Save the DataFrame
    df.to_csv('captchas.csv')

    # Choose a random index from the DataFrame
    random_index = random.randint(0, len(df) - 1)

    # Get the captcha image bytes and path from the selected index
    captcha_image_bytes = df.loc[random_index, 'image']['bytes']
    captcha_image_path = df.loc[random_index, 'image']['path']

    # Create a directory to save the captcha images if it doesn't exist
    save_directory = 'saved_captchas'
    os.makedirs(save_directory, exist_ok=True)

    # Save the captcha image to a file
    with open(os.path.join(save_directory, captcha_image_path), 'wb') as image_file:
        image_file.write(captcha_image_bytes)

    print(f"Captcha image saved at: {os.path.join(save_directory, captcha_image_path)}")

