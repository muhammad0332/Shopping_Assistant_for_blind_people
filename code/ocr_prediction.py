from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import cv2
import re

model_ocr = ocr_predictor(pretrained=True)

def extract_text_prices_and_dates(frame):
    image_path = 'frame.jpg'
    cv2.imwrite(image_path, frame)
    single_img_doc = DocumentFile.from_images(image_path)
    ocr_result = model_ocr(single_img_doc)

    extracted_texts = []
    prices = []
    mfg_date = None
    exp_date = None

    currency_patterns = [r'\b[₹$PKRRSrs]+\s*[\d,]+\.?\d*\b']

    capture_mfg = False
    capture_exp = False
    mfg_parts = []
    exp_parts = []

    for page in ocr_result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text = word.value
                    extracted_texts.append(text)

                    for pattern in currency_patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            prices.append(text)

                    if "MFG"  in text.upper():
                        capture_mfg = True
                        continue

                    if "EXP"  in text.upper():
                        capture_exp = True
                        continue

                    if capture_mfg:
                        mfg_parts.append(text)
                        if len(mfg_parts) == 3: 
                            mfg_date = " ".join(mfg_parts)
                            capture_mfg = False

                    if capture_exp:
                        exp_parts.append(text)
                        if len(exp_parts) == 3:  
                            exp_date = " ".join(exp_parts)
                            capture_exp = False

    return extracted_texts, prices, mfg_date, exp_date
