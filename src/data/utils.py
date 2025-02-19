import re


def anonymize_text(text: str) -> str:
    """
    Anonymizes sensitive information in text by replacing it with generic placeholders.

    This function replaces various types of sensitive information including:
    - Names (including titles like Mr., Mrs., Dr.)
    - URLs and domain names
    - Dates in various formats
    - Social Security Numbers
    - Credit card numbers
    - Passwords
    - Generic numbers

    Args:
        text (str): The input text to be anonymized

    Returns:
        str: The anonymized text with sensitive information replaced by placeholders

    Examples:
        >>> anonymize_text("Mr. John Smith accessed example.com on 2024-03-15")
        '[Name] accessed [DOMAIN] on [DATE]'
    """

    # Handle other potential names
    def replace_names(match):
        word = match.group(0)
        # Skip if it's the first word in a sentence or after a period
        prev_char = text[max(0, match.start() - 2) : match.start()].strip()
        if not prev_char or prev_char.endswith("."):
            return word
        # If it passes all checks, it's probably a name
        return "[Name]"

    # Replace single capitalized words that might be names
    text = re.sub(r"\b[A-Z][a-z]+\b", replace_names, text)
    # Replace names (assuming they follow common patterns)
    text = re.sub(r"(?:Mr\.|Mrs\.|Ms\.|Dr\.) [A-Z][a-z]+ [A-Z][a-z]+", "[Name]", text)
    text = re.sub(
        r"[A-Z][a-z]+ [A-Z][a-z]+(?=\s(?:requested|changed|accessed|approved))",
        "[Name]",
        text,
    )

    # Replace URLs and domains
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"\b[\w-]+\.(?:com|org|net|biz|info)\b", "[DOMAIN]", text)

    # Replace dates
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "[DATE]", text)
    text = re.sub(r"\d{2}/\d{2}/\d{4}", "[DATE]", text)
    text = re.sub(
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2},\s+\d{4}",
        "[DATE]",
        text,
    )

    # Replace common numeric patterns
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)  # SSN pattern
    text = re.sub(r"\b\d{16}\b", "[CARD_NUMBER]", text)  # Credit card number pattern
    text = re.sub(
        r"\b(?:password|pwd):\s*\S+", "[PASSWORD]", text, flags=re.IGNORECASE
    )  # Passwords
    text = re.sub(r"\b\d+\b", "[NUMBER]", text)  # Any remaining numbers

    # Clean up any multiple spaces created during replacements
    text = re.sub(r"\s+", " ", text).strip()

    return text
