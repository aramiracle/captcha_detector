import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from math import sin, radians
from torchvision import transforms
from io import BytesIO
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

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
    draw_lines(draw, width, height, 25)

    # Convert image to RGBA before applying elastic transformation
    captcha = captcha.convert('RGBA')

    # Apply random elastic transformation
    captcha = elastic_transform(captcha, alpha=70.0, sigma=10.0)

    # Convert image back to RGB after elastic transformation
    captcha = captcha.convert('RGB')

    # Resize to half size to save memory
    captcha = captcha.resize((width // 3, height // 3))

    # Apply image smoothing
    captcha = captcha.filter(ImageFilter.SMOOTH)

    # Generate a file name based on the captcha text
    file_name = f"{captcha_text}.png"

    # Convert image to binary bytes
    image_bytes = BytesIO()
    captcha.save(image_bytes, format='PNG')
    image_bytes.seek(0)  # Reset the buffer position to the beginning

    # Return a dictionary with bytes and file_name as a string
    return image_bytes.getvalue(), file_name, f"This is '{captcha_text}'"

def generate_captcha_data(num_captchas):
    captcha_data = []

    for i in tqdm(range(num_captchas), desc="Generating Captchas"):
        # Generate captcha
        captcha_bytes, file_name, text = generate_captcha()

        # Append captcha data to the list
        captcha_data.append({'image.bytes': captcha_bytes, 'image.path':file_name, 'text': text})

    return captcha_data

def save_grid_image(captchas_df, rows=5, cols=20, output_dir='saved_captchas_grid'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = []

    for _, captcha in captchas_df.iterrows():
        image_bytes = captcha['image.bytes']
        image = Image.open(BytesIO(image_bytes))
        images.append(image)

    max_height = max(image.height for image in images)
    combined_width = max(image.width for image in images) * cols

    grid_image = Image.new('RGB', (combined_width, max_height * rows), color='white')
    current_width, current_height = 0, 0

    for image in images:
        grid_image.paste(image, (current_width, current_height))
        current_width += image.width

        if current_width >= combined_width:
            current_width = 0
            current_height += max_height

    grid_image_path = os.path.join(output_dir, 'grid_captchas.png')
    grid_image.save(grid_image_path)

    print(f"Grid image saved at: {grid_image_path}")

if __name__ == "__main__":
    num_captchas = 200
    filename = 'captchas.parquet'
    captcha_data = generate_captcha_data(num_captchas)

    # Create Arrow Table from captcha data
    schema = pa.schema([
        ('image.bytes', pa.binary()),
        ('image.path', pa.string()),
        ('text', pa.string())
    ])
    arrow_data = [
        pa.array([item['image.bytes'] for item in captcha_data]),
        pa.array([item['image.path'] for item in captcha_data]),
        pa.array([item['text'] for item in captcha_data])
    ]
    arrow_table = pa.Table.from_arrays(arrow_data, schema=schema)

    # Save the Arrow Table as Parquet
    pq.write_table(arrow_table, filename)

    # Adjust rows and cols for the grid image
    rows, cols = 20, 5
    save_grid_image(arrow_table.to_pandas(), rows=rows, cols=cols)
