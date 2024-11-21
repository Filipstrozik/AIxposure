# AIxposure

AIxposure is a Python script tool to detect AI-generated content in social media. This tool was developed as part of the course "Analysis of " at the Wrocław University of Science and Technology. AI Department.

## Installation

### Tesseract requirement

To install AIxposure, you need to have Tesseract OCR installed on your local machine. You can install Tesseract OCR using the following instructions:

### macOS

```sh
brew install tesseract
```

### Ubuntu

```sh
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### Windows

Download the installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and follow the installation instructions.

After installing Tesseract OCR, verify the installation by running:

```sh
tesseract --version
```

## Download AIxposure

### Executable application

You can download the executable application from the artifacts in the Releases section.

### Source code

You can clone the repository using the following command:

```sh
git clone https://github.com/Filipstrozik/AIxposure.git
```

## Usage

### Executable application

Just run the executable application and follow the instructions. Give all the necessary permissions to the application, specialy the access of input files.

### Source code

Install the required Python packages using the following command:

```sh
pip install -r requirements.txt
```

In this case you can use virtual environment to install the required packages.

To run the source code, you need to have Python 3 installed on your local machine. You can install Python 3 using the following instructions:

```sh
python3 aixposure.py
```

### Succesfull Run

If the application runs successfully, you should see the following output:

```sh
Loading the text and image classification models...
Models loaded successfully!
Welcome to the 'AIxposure' application!
This application allows you to take a screenshot and analyze it using AI models to detect AI-generated and human-written content.
--------------------------------------------
Press the following shortcuts to take a screenshot:
Press <ctrl>+<shift>+s to take a screenshot of both text and image
Press <ctrl>+<shift>+t to take a screenshot of only text
Press <ctrl>+<shift>+i to take a screenshot of only image
Press <ctrl>+<esc> to quit the program
```

## Shortcuts

Use the following shortcuts to take screenshots and analyze content for AI generation:

- **CTRL + SHIFT + S**: Detect both text and image to determine if they are AI-generated.
- **CTRL + SHIFT + T**: Detect only text to determine if it is AI-generated.
- **CTRL + SHIFT + I**: Detect only images to determine if they are AI-generated.
- **CTRL + ESC**: Quit the program.

# References

- RADAR (Text detection model):

  - https://github.com/IBM/RADAR
  - "RADAR: Robust AI-Text Detection via Adversarial Learning" Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho https://arxiv.org/abs/2307.03838

- ORGANIKA SDXL (Image detection model):
  - https://huggingface.co/Organika
  - https://huggingface.co/Organika/sdxl-detector
