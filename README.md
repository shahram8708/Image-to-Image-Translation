# Image-to-Image Translation with Pix2Pix

This project implements an image-to-image translation model using a conditional generative adversarial network (cGAN) called Pix2Pix. It allows users to upload an image, which is then translated into a different style or representation using a pre-trained Pix2Pix model.

## Features

- **Upload Image:** Users can upload an image to be translated.
- **Image Translation:** The uploaded image is processed using a Pix2Pix model to generate a translated image.
- **Download Result:** Users can download the translated image after processing.

## Technologies Used

- **Python:** Programming language used for backend logic.
- **Flask:** Web framework used for building the application.
- **TensorFlow:** Deep learning library used to implement the Pix2Pix model.
- **Pillow:** Python Imaging Library (PIL) used for image processing tasks.
- **HTML/CSS/JavaScript:** Frontend development for user interface and interactions.

## Setup

### Prerequisites

- Python 3.7 or higher
- Pip package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shahram8708/Image-to-Image-Translation.git
   ```

2. Navigate into the project directory:

   ```bash
   cd Image to Image Translation
   ```

3. Install required Python packages using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Start the Flask application:

   ```bash
   python app.py
   ```

2. Open a web browser and go to `http://localhost:5000` to access the application.

3. Upload an image using the provided interface.

4. Wait for the image translation process to complete.

5. Download the translated image from the result section.

### Folder Structure

- **app.py:** Main Flask application file containing route definitions and server configuration.
- **templates/:** HTML templates for frontend rendering.
- **static/:** Static files (CSS, JavaScript) for frontend styling and functionality.
- **uploads/:** Directory to store uploaded images.
- **results/:** Directory to store translated images.

### Credits

- This project utilizes the Pix2Pix model architecture based on the original paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) by Phillip Isola et al.

### License

- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
