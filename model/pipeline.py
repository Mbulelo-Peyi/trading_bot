import pandas as pd
import re
from textblob import TextBlob
from tensorflow.keras.utils import to_categorical

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", " <NUM> ", text)
    text = re.sub(r"\$", " <DOLLAR> ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def has_keywords(text, keywords):
    return int(any(word in text.lower() for word in keywords))

# Load data
df = pd.read_csv("data.csv")
df = df[['Sentence', 'Sentiment']].dropna()
df['clean_text'] = df['Sentence'].apply(clean_text)

# Encode label
label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
df['Sentiment'] = df['Sentiment'].map(label_map)

# Add features
df['char_count'] = df['Sentence'].apply(len)
df['word_count'] = df['Sentence'].apply(lambda x: len(str(x).split()))
df['avg_word_len'] = df['char_count'] / (df['word_count'] + 1)
df['has_buy'] = df['Sentence'].apply(lambda x: has_keywords(x, ['buy', 'long', 'entry']))
df['has_sell'] = df['Sentence'].apply(lambda x: has_keywords(x, ['sell', 'short', 'exit']))
df['has_price'] = df['Sentence'].apply(lambda x: int(bool(re.search(r"\d+\.*\d*", x))))
df['has_pct'] = df['Sentence'].apply(lambda x: int('%' in x))
df['has_stock'] = df['Sentence'].apply(lambda x: int(bool(re.search(r"[A-Z]{2,5}", x))))
df['sentiment_polarity'] = df['Sentence'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment_subjectivity'] = df['Sentence'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Prepare final features
text_data = df['clean_text'].values
structured_data = df[['char_count', 'word_count', 'avg_word_len', 'has_buy',
                      'has_sell', 'has_price', 'has_pct', 'has_stock',
                      'sentiment_polarity', 'sentiment_subjectivity']].values
labels = to_categorical(df['Sentiment'], num_classes=3)
