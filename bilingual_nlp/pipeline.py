
from language_detection import detect_language
from preprocessing import preprocess_text
from hinglish_converter import normalize_hinglish
from claim_extraction import extract_claim


def process_claim(text):

    # Step 1: Language Detection
    language = detect_language(text)

    # Step 2: Text Preprocessing
    cleaned_text = preprocess_text(text)

    # Step 3: Hinglish Normalization
    if language == "Hinglish":
        normalized_text = normalize_hinglish(cleaned_text)
    else:
        normalized_text = cleaned_text

    # Step 4: Claim Extraction
    claim = extract_claim(normalized_text)

    return {
        "original_text": text,
        "language": language,
        "cleaned_text": cleaned_text,
        "normalized_text": normalized_text,
        "extracted_claim": claim
    }


# Main Program
if __name__ == "__main__":

    user_input = input("Enter your claim: ")

    result = process_claim(user_input)

    print("\n----- VERILENS NLP OUTPUT -----")

    for key, value in result.items():
        print(key, ":", value)
