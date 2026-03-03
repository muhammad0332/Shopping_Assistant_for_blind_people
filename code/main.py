from speechRecognizer import capture_audio_from_microphone
from live_capturing import capture_image_from_camera
from yolo_prediction import detect_products
import matplotlib.pyplot as plt
from text_speech import text_to_speech
from ultralytics import YOLO
from OCR_pytrassactlib import extract_text_and_dates
def main():
    print("Please speak the product name.")
    spoken_class = capture_audio_from_microphone()
    if not spoken_class:
        print("No valid class detected from voice input. Exiting...")
        return

    print(f"Now, show the {spoken_class} product to the camera.")
    captured_image = capture_image_from_camera()

    if captured_image is not None:
        model= YOLO("D:/updated_code/models/yolov8/yolov8_model.pt")
        processed_image, keyword_found, class_counts = detect_products(captured_image, spoken_class, model)
        # extracted_texts, prices, mfg_date, exp_date = extract_text_prices_and_dates(processed_image)
        extracted_texts, prices, mfg_date, exp_date, processed_image = extract_text_and_dates(captured_image, use_grayscale=True)
        print("\nExtracted Text:")
        for text in extracted_texts:
           print(f"Text: {text}")


        if prices:
            print("\nIdentified Prices in this frame:")
            
            print(f"Price: {prices}")
        else:
            print("No prices identified in this frame.")


        if mfg_date:
            print(f"\nManufacturing Date: {mfg_date}")
        else:
            print("No MFG date identified in this frame.")

        if exp_date:
            print(f"Expiration Date: {exp_date}")
        else:
            print("No Expiration date identified in this frame.")


        if keyword_found:
            result_message = f"Successfully! Detected '{spoken_class}' in the image."
            print(result_message)
            text_to_speech(result_message) 
            print(f"Detected products count: {class_counts}")
        else:
            result_message = f"'{spoken_class}' was not detected in the image."
            print(result_message)
            text_to_speech(result_message) 
        plt.imshow(processed_image)
        plt.title(f"Detection results for '{spoken_class}'")
        plt.axis('off')
        plt.show()
    else:
        print("No image captured.")

if __name__ == '__main__':
    main()
