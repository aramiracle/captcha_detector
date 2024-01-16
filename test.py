from PIL import Image
import pytesseract

def captcha_detection(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Use pytesseract to perform OCR on the grayscale image
    captcha_text = pytesseract.image_to_string(gray_image)

    return captcha_text

if __name__ == "__main__":
    # Replace 'path/to/your/captcha/image.png' with the actual path to your captcha image
    captcha_image_path = 'image1.png'

    try:
        detected_text = captcha_detection(captcha_image_path)
        print(f'Detected Captcha Text: {detected_text}')
    except Exception as e:
        print(f'Error: {e}')
