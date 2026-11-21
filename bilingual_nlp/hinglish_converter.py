
# Hinglish to Hindi Normalization

hinglish_dictionary = {
    "kisan": "किसान",
    "kisans": "किसान",
    "sarkar": "सरकार",
    "yojana": "योजना",
    "paisa": "पैसा",
    "paise": "पैसे",
    "milega": "मिलेगा",
    "milenge": "मिलेंगे",
    "nahi": "नहीं",
    "hai": "है",
    "hain": "हैं",
    "ko": "को",
    "mein": "में",
    "ka": "का",
    "ki": "की",
    "ke": "के"
}


def normalize_hinglish(text):

    words = text.lower().split()

    normalized_words = []

    for word in words:
        normalized_word = hinglish_dictionary.get(word, word)
        normalized_words.append(normalized_word)

    return " ".join(normalized_words)


# Testing
if __name__ == "__main__":

    text = "kisan ko sarkar ki yojana mein paisa milega"

    result = normalize_hinglish(text)

    print("Original:", text)
    print("Normalized:", result)
