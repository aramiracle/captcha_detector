import gradio as gr
from PIL import Image
import numpy as np


# Function to perform captcha detection using the model
def detect_captcha(image):
    # Preprocess the image (adjust according to your model's requirements)
    image = Image.fromarray(image.astype('uint8'))
    # Your preprocessing steps here...

    # Convert the image to a format suitable for model input
    input_data = np.array(image)  # Adjust if needed

    # Perform prediction using your captcha model
    predicted_text = captcha_model.predict(input_data)

    return predicted_text

# Create a Gradio interface
iface = gr.Interface(
    fn=detect_captcha,
    inputs=gr.Image(),
    outputs=gr.Textbox(),
    live=True,
    capture_session=True,  # Capture the input for debugging
    title="Captcha Detection",
    description="Upload an image with a captcha to detect the text.",
)

# Launch the interface
iface.launch()
