import threading
import tkinter as tk
from pynput import keyboard
from PIL import Image, ImageTk
import pytesseract
from typing import Tuple
import cv2
import numpy as np
import pyautogui
from transformers import pipeline
import os
import torch
from enum import Enum

if torch.cuda.is_available():
    device = 0  # CUDA device
elif torch.backends.mps.is_available():
    device = 0  # MPS device
else:
    device = -1  # CPU

text_and_image_shortcut = "<ctrl>+<shift>+s"
only_text_shortcut = "<ctrl>+<shift>+t"
only_image_shortcut = "<ctrl>+<shift>+i"
quit_shortcut = "<ctrl>+<esc>"


class ShortcutOption(Enum):
    TEXT_AND_IMAGE = 1
    TEXT = 2
    IMAGE = 3


# load the text and image classification models
print("Loading the text and image classification models...")

# Disable parallelism in tokenizers to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# "RADAI"
""" 
@misc{hu2023radar,
      title={RADAR: Robust AI-Text Detection via Adversarial Learning},
      author={Xiaomeng Hu and Pin-Yu Chen and Tsung-Yi Ho},
      year={2023},
      eprint={2307.03838},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""
text_pipe = pipeline(
    "text-classification", model="TrustSafeAI/RADAR-Vicuna-7B", device=device
)
# "Organika/sdxl-detector"
# umm-maybe/AI-image-detector
"""
https://huggingface.co/umm-maybe/AI-image-detector
"""
img_pipe = pipeline(
    "image-classification", model="Organika/sdxl-detector", device=device
)

print("Models loaded successfully!")

extracted_image = None
extracted_text = None


def crop_image(image: Image.Image, option: ShortcutOption):

    global extracted_image, extracted_text
    # Open the screenshot in a new window
    top = tk.Toplevel()
    top.title("Crop Image")
    top.attributes("-fullscreen", True)
    canvas = tk.Canvas(top, width=image.width, height=image.height)
    canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

    top.update_idletasks()

    # Resize the image to fit the canvas with a margin
    img_width, img_height = image.size
    canvas_width = canvas.winfo_screenwidth()
    canvas_height = canvas.winfo_screenheight()

    scale = min(canvas_width / img_width, canvas_height / img_height)
    new_size = (int(img_width * scale), int(img_height * scale))
    resized_image = image.resize(new_size, Image.LANCZOS)

    img = ImageTk.PhotoImage(resized_image)
    canvas.image = img
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

    # Variables to store the cropping coordinates
    rect = None
    start_x = start_y = None

    def on_button_press(event):
        nonlocal start_x, start_y, rect
        start_x, start_y = event.x, event.y
        rect = canvas.create_rectangle(
            start_x, start_y, start_x, start_y, outline="red", dash=(4, 2)
        )

    def on_move_press(event):
        nonlocal rect
        cur_x, cur_y = event.x, event.y
        canvas.coords(rect, start_x, start_y, cur_x, cur_y)

    def extract_text(screenshot: Image.Image):
        extracted_text = pytesseract.image_to_string(screenshot)
        return extracted_text

    def extract_image(screenshot: Image.Image) -> Tuple[Image.Image, Image.Image]:
        def get_area(image):
            if image is not None:
                return image.shape[0] * image.shape[1]
            else:
                return 0

        # Load the screenshot image as numpy array
        screenshot_arr = np.array(screenshot)
        screenshot_area = get_area(screenshot_arr)

        # Preprocess the image (resize, convert to grayscale, etc.)
        gray = cv2.cvtColor(screenshot_arr, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Extract largest image
        largest_image = None
        screenshot_redacted = None
        for contour in contours:
            # Get bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Crop image to the bounding box
            extracted_image = screenshot_arr[y : y + h, x : x + w]
            extracted_area = w * h

            if extracted_area < screenshot_area and extracted_area > get_area(
                largest_image
            ):
                largest_image = extracted_image
                screenshot_redacted = screenshot_arr.copy()
                screenshot_redacted[y : y + h, x : x + w] = 0

        # Convert numpy array back to PIL Image
        extracted_image = Image.fromarray(largest_image)
        screenshot_redacted = Image.fromarray(screenshot_redacted)

        return screenshot_redacted, extracted_image

    def on_button_release(event) -> Tuple[Image.Image, str]:
        global extracted_image, extracted_text
        nonlocal rect, start_x, start_y
        end_x, end_y = event.x, event.y
        start_x_orig = int(start_x / scale)
        start_y_orig = int(start_y / scale)
        end_x_orig = int(end_x / scale)
        end_y_orig = int(end_y / scale)

        cropped = image.crop((start_x_orig, start_y_orig, end_x_orig, end_y_orig))

        if option == ShortcutOption.TEXT:
            extracted_text = extract_text(cropped)
        elif option == ShortcutOption.IMAGE: # cropped image should only contain image to detect.
            extracted_image = cropped
        else:
            redacted, extracted_image = extract_image(cropped)
            extracted_text = extract_text(redacted)

        top.destroy()

        use_extracted_image_and_text(option)

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)

    def on_key_press(event):
        if event.keysym == "Escape":
            top.destroy()

    top.bind("<KeyPress>", on_key_press)


def use_extracted_image_and_text(option: ShortcutOption):
    global extracted_image, extracted_text

    text_result, img_result = None, None
    text_label_map = {"LABEL_0": "AI-generated", "LABEL_1": "Human-written"}
    img_label_map = {"human": "Human", "artificial": "AI"}

    if option in (ShortcutOption.TEXT, ShortcutOption.TEXT_AND_IMAGE):
        text_classification_result = text_pipe(extracted_text)
        text_result = (
            f"Text is likely: {text_label_map[text_classification_result[0]['label']]} "
            f"with a confidence score of {text_classification_result[0]['score']:.2f}"
        )

    # Process image classification if only IMAGE or both IMAGE and TEXT are requested
    if option in (ShortcutOption.IMAGE, ShortcutOption.TEXT_AND_IMAGE):
        img_classification_result = img_pipe(extracted_image)
        img_result = (
            f"Image is likely: {img_label_map[img_classification_result[0]['label']]} "
            f"with a confidence score of {img_classification_result[0]['score']:.2f}"
        )

    # Setup the popup window
    widget_width, widget_height = 400, 400
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    x_pos, y_pos = screen_width - widget_width, screen_height - widget_height

    popup = tk.Toplevel()
    popup.title("AIxposure Results")
    popup.geometry(f"{widget_width}x{widget_height}+{x_pos}+{y_pos}")

    # Show text widget if TEXT or both IMAGE and TEXT are requested
    if option in (ShortcutOption.TEXT, ShortcutOption.TEXT_AND_IMAGE):
        text_label = tk.Label(popup, text=text_result)
        text_label.pack()
        text_widget = tk.Text(popup, height=10)
        text_widget.insert(tk.END, extracted_text)
        text_widget.pack(expand=True, fill=tk.BOTH)

    # Show image widget if IMAGE or both IMAGE and TEXT are requested
    if option in (ShortcutOption.IMAGE, ShortcutOption.TEXT_AND_IMAGE):
        img_label = tk.Label(popup, text=img_result)
        img_label.pack()
        img_widget = tk.Label(popup)

        # Resize and display the image
        img_width, img_height = extracted_image.size
        scale = widget_width / img_width
        resized_image = extracted_image.resize(
            (int(img_width * scale), int(img_height * scale)), Image.LANCZOS
        )
        img = ImageTk.PhotoImage(resized_image)
        img_widget.configure(image=img)
        img_widget.image = img
        img_widget.pack()

    popup.update_idletasks()


def take_screenshot():
    screenshot = pyautogui.screenshot()
    return screenshot


def on_screenshot_shortcut(option: ShortcutOption):
    # reset the global variables
    global extracted_image, extracted_text
    extracted_image = None
    extracted_text = None

    screenshot = take_screenshot()
    crop_image(screenshot, option)


# Function to close the application
def quit_program():
    print("Quitting program...")
    root.quit()  # Stop the tkinter main loop


# Function to listen for keyboard shortcuts
def listen_for_shortcuts():
    def on_text_and_image():
        print(f"{text_and_image_shortcut} detected!")
        root.after(0, on_screenshot_shortcut, ShortcutOption.TEXT_AND_IMAGE)

    def on_text():
        print(f"{only_text_shortcut} detected!")
        root.after(0, on_screenshot_shortcut, ShortcutOption.TEXT)

    def on_image():
        print(f"{only_image_shortcut} detected!")
        root.after(0, on_screenshot_shortcut, ShortcutOption.IMAGE)

    def on_activate_quit():
        print(f"{quit_shortcut} detected!")
        root.after(0, quit_program)

    # Define shortcuts
    with keyboard.GlobalHotKeys(
        {
            text_and_image_shortcut: on_text_and_image,
            only_text_shortcut: on_text,
            only_image_shortcut: on_image,
            quit_shortcut: on_activate_quit,
        }
    ) as listener:
        listener.join()


def main():
    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=listen_for_shortcuts)
    listener_thread.daemon = True
    listener_thread.start()

    # print shortcuts
    print("Welcome to the 'AIxposure' application!")
    print(
        "This application allows you to take a screenshot and analyze it using AI models to detect AI-generated and human-written content."
    )
    print("--------------------------------------------")
    print("Press the following shortcuts to take a screenshot:")
    print(
        f"Press {text_and_image_shortcut} to take a screenshot of both text and image"
    )
    print(f"Press {only_text_shortcut} to take a screenshot of only text")
    print(f"Press {only_image_shortcut} to take a screenshot of only image")
    print(f"Press {quit_shortcut} to quit the program")

    # Main tkinter window
    global root
    root = tk.Tk()
    root.withdraw()
    root.mainloop()


if __name__ == "__main__":
    main()
