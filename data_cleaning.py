def clean_document(text: str) -> str:
    text = text[int(len(text) * 0.3) :]
    return text
