from speechRecognizer import capture_audio_from_microphone
from live_capturing import capture_image_from_camera
from yolo_prediction import detect_products
import matplotlib.pyplot as plt
from ocr_prediction import extract_text_prices_and_dates
from ultralytics import YOLO


# Main function to capture voice and image, and make predictions
def main():
    print("Please speak the product name.")
    spoken_class = capture_audio_from_microphone()
    if not spoken_class:
        print("No valid class detected from voice input. Exiting...")
        return

    print(f"Now, show the {spoken_class} product to the camera.")
    captured_image = capture_image_from_camera()

    if captured_image is not None:
        # Load YOLO model
        model= YOLO("models/yolov8/yolov8_model.pt")

        # Detect products
        processed_image, keyword_found, class_counts = detect_products(captured_image, spoken_class, model)
        extracted_texts, prices, mfg_date, exp_date = extract_text_prices_and_dates(processed_image)

        print("\nExtracted Text:")
        for text in extracted_texts:
           print(f"Text: {text}")


        if prices:
            print("\nIdentified Prices in this frame:")
            for price in prices:
                print(f"Price: {price}")
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


        # Check if the keyword was found in the image
        if keyword_found:
            print(f"Success! Detected '{spoken_class}' in the image.")
            print(f"Detected products count: {class_counts}")
        else:
            print(f"'{spoken_class}' was not detected in the image.")

        # Display the result image
        plt.imshow(processed_image)
        plt.title(f"Detection results for '{spoken_class}'")
        plt.axis('off')
        plt.show()
    else:
        print("No image captured.")

if __name__ == '__main__':
    main()
