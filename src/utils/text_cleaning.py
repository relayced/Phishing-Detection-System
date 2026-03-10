import re
from bs4 import BeautifulSoup

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ")


def normalize_urls(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return URL_PATTERN.sub(" [URL] ", text)


def clean_email_text(text: str) -> str:
    text = strip_html(text)
    text = normalize_urls(text)
    text = text.lower()
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def extract_first_url(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = URL_PATTERN.search(text)
    return match.group(0) if match else ""
