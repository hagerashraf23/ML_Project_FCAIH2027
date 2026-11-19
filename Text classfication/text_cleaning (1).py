import re
import html
import unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

extra_stopwords = ['aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'll', 'mustn', 'shouldn', 've', 'wasn', 'weren', 'wouldn']
stop_words = set(ENGLISH_STOP_WORDS) | set(extra_stopwords)

def clean_text(title, content):
    # Step 1: Combine title + space + content
    text = str(title) + ' ' + str(content)

    # Step 2: Normalize extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 3: Decode HTML entities
    text = html.unescape(text)

    # Step 4: Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)

    # Step 5: Remove URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # Step 6: Remove accents and non-ASCII characters
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Step 7: Convert to lowercase
    text = text.lower()

    # Step 8: Replace anything other than a-z with spaces
    text = re.sub(r'[^a-z]', ' ', text)

    # Step 9: Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Step 10: Remove words shorter than 2 characters
    words = [word for word in words if len(word) >= 2]

    # Normalize spaces
    return ' '.join(words)


def clean_single(text):
    # For the GUI: one text typed by the user (there is no separate title)
    return clean_text('', text)
