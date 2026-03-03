import speech_recognition as sr

classes = [
    "cooking oil",
    "dishwash liquid",
    "fruit jam",
    "lays chips",
    "shampoo",
    "soya sauce",
    "toilet cleaner",
    "tomato ketchup",
    "dettol liquid",
    "tea"
]

def match_class_from_text(text):
    for class_name in classes:
        if class_name in text:
            return class_name
    return None

def capture_audio_from_microphone():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        r.adjust_for_ambient_noise(source, duration=1)

        print("Listening for your input...")
        audio = r.listen(source) 

    try:
        recognized_text = r.recognize_google(audio).lower()  
        print("Recognized text: " + recognized_text)
        matched_class = match_class_from_text(recognized_text)
        if matched_class:
            print(f"Matched class: {matched_class}")
            return matched_class
        else:
            print("No matching class found.")
            return None
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    
 