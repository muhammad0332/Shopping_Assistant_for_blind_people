import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
import re
import cv2
from dateutil.parser import parse


def preprocess_image(image_path, img_size=(224, 224)):
    """ Preprocess image for model input """
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def sharpen_image(image):
    """ Sharpen the image to enhance OCR accuracy """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def denoise_image(image):
    """ Apply denoising to remove noise from the image """
    return cv2.fastNlMeansDenoising(image, h=20)

def preprocess_for_ocr(image, use_grayscale=True):
    """ Preprocess image: optional grayscale, resize, sharpen, and threshold """
    if use_grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    image = sharpen_image(image)
    return image

def extract_valid_date(text):
    """ Extract a valid date from text """
    try:
        return parse(text, fuzzy=True).strftime('%d-%b-%Y')
    except ValueError:
        return None

def is_valid_price(price):
    currency_symbols = ['\u20B9', '\$', '€', '£', 'PKR', 'RS', 'rs','Rs']
    if any(symbol.lower() in price.lower() for symbol in currency_symbols):
        try:

            value = float(re.sub(r'[^\d.]', '', price))
            return value > 10  
        except ValueError:
            return False
    return False
def extract_text_and_dates(image, use_grayscale=True):
    if image is None:
        raise ValueError("Image not found or OpenCV cannot read it.")
    processed_image = preprocess_for_ocr(image, use_grayscale=use_grayscale)
    custom_config = r'--oem 3 --psm 6'
    ocr_result = pytesseract.image_to_string(processed_image, config=custom_config)
    extracted_texts = ocr_result.split("\n")
    prices = []
    mfg_date = None
    exp_date = None
    price_pattern = r'\b(?:[\u20B9$€£PKR]|(?i)RS)\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b'
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{2,4})\b'

    for line in extracted_texts:
        line_upper = line.upper()
        if re.search(price_pattern, line, flags=re.IGNORECASE):
            detected_prices = re.findall(price_pattern, line, flags=re.IGNORECASE)
            valid_prices = [p for p in detected_prices if is_valid_price(p)]
            prices.extend(valid_prices)
        if "MFG" in line_upper and not mfg_date:
            date_match = re.search(date_pattern, line)
            if date_match:
                mfg_date = extract_valid_date(date_match.group())
        if "EXP" in line_upper and not exp_date:
            date_match = re.search(date_pattern, line)
            if date_match:
                exp_date = extract_valid_date(date_match.group())

    return extracted_texts, prices, mfg_date, exp_date, processed_image

