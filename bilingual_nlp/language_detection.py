
import re

def detect_language(text):

    if not text or not text.strip():
        return "Empty"

    # Detect Hindi characters
    hindi_pattern = re.search(r'[\u0900-\u097F]', text)

    if hindi_pattern:
        if re.search(r'[a-zA-Z]', text):
            return "Mixed"
        return "Hindi"

    # Detect common Hinglish words
    hinglish_words = [
        "hai", "hain", "kya", "kaise",
        "mein", "ko", "ka", "ki",
        "nahi", "milenge", "sarkar"
    ]

    words = text.lower().split()

    if any(word in hinglish_words for word in words):
        return "Hinglish"

    return "English"


# Testing
print(detect_language("Farmers will receive benefits"))
print(detect_language("किसानों को लाभ मिलेगा"))
print(detect_language("Farmers ko benefits milenge"))
