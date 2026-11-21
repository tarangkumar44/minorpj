
import re

def extract_claim(text):

    if not text or not text.strip():
        return ""

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove common question phrases
    text = re.sub(
        r'^(kya ye sach hai ki|is it true that|kya)\s+',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Remove question mark
    text = text.rstrip("?! ")

    # Remove trailing question word
    text = re.sub(
        r'\s+kya$',
        '',
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


# Testing
if __name__ == "__main__":

    text = "PM Kisan scheme me farmers ko 8000 rupees milte hain kya?"

    claim = extract_claim(text)

    print("Original Input:", text)
    print("Extracted Claim:", claim)
