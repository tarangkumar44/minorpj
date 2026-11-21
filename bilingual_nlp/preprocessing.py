
import re
import unicodedata

def preprocess_text(text):

    # Check empty input
    if not text or not text.strip():
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Convert English characters to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove unnecessary punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra spaces again
    text = text.strip()

    return text


# Testing the function
if __name__ == "__main__":

    text1 = "FARMERS KO BENEFIT MILEGA!!!"
    text2 = "सरकार   ने योजना शुरू की"
    text3 = "PM KISAN  8000 RUPEES"

    print("Original:", text1)
    print("Processed:", preprocess_text(text1))

    print("\nOriginal:", text2)
    print("Processed:", preprocess_text(text2))

    print("\nOriginal:", text3)
    print("Processed:", preprocess_text(text3))
