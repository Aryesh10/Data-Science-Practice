import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from textstat import textstat
from textblob import TextBlob
import nltk

nltk.download('punkt')
nltk.download('stopwords')

input_path = 'Input.xlsx'
output_path = 'Output Data Structure.xlsx'
output_csv = 'output_results.csv'
output_folder = 'extracted_articles'

input_data = pd.read_excel(input_path)
output_template = pd.read_excel(output_path)

os.makedirs(output_folder, exist_ok=True)

try:
    with open('Master Dictionary/positive-words.txt', 'r', encoding='latin-1') as f:
        positive_words = set(f.read().split())
    with open('Master Dictionary/negative-words.txt', 'r', encoding='latin-1') as f:
        negative_words = set(f.read().split())
except Exception as e:
    print(f"Error loading positive or negative words: {e}")
    positive_words, negative_words = set(), set()

stop_words = set(stopwords.words('english'))
stopwords_folder = 'Stop Words'

if not os.path.exists(stopwords_folder):
    print(f"Warning: Stop Words folder not found. Proceeding with default NLTK stop words.")
else:
    for stopword_file in os.listdir(stopwords_folder):
        try:
            file_path = os.path.join(stopwords_folder, stopword_file)
            with open(file_path, 'r', encoding='latin-1') as f:
                stop_words.update(f.read().split())
        except Exception as e:
            print(f"Error reading file {stopword_file}: {e}")


def clean_text(text):
    """Cleans text by removing extra spaces and stop words."""
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    filtered_words = [word for word in cleaned_text.split() if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def calculate_positive_negative_scores(text):
    """Calculates the positive and negative scores after cleaning."""
    words = text.split()
    positive_score = sum(1 for word in words if word.lower() in positive_words)
    negative_score = sum(1 for word in words if word.lower() in negative_words)
    return positive_score, negative_score

def calculate_polarity_score(positive_score, negative_score):
    """Calculates polarity score."""
    total = positive_score + negative_score
    return (positive_score - negative_score) / total if total > 0 else 0

def calculate_subjectivity_score(text):
    """Calculates subjectivity score."""
    return TextBlob(text).sentiment.subjectivity

def calculate_complex_word_count(text):
    """Counts complex words (words with >= 3 vowels)."""
    words = text.split()
    return sum(1 for word in words if len(re.findall(r'[aeiou]', word.lower())) >= 3)

def calculate_avg_word_length(text):
    """Calculates average word length."""
    words = text.split()
    return sum(len(word) for word in words) / len(words) if words else 0

results = []
for _, row in input_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('title').get_text(strip=True) if soup.find('title') else "No Title"
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        article_text = clean_text(article_text)

        if not article_text:
            raise ValueError("No valid article content found.")

        article_path = os.path.join(output_folder, f"{url_id}.txt")
        with open(article_path, 'w', encoding='utf-8') as file:
            file.write(f"{title}\n{article_text}")

        positive_score, negative_score = calculate_positive_negative_scores(article_text)
        polarity_score = calculate_polarity_score(positive_score, negative_score)
        subjectivity_score = calculate_subjectivity_score(article_text)
        avg_sentence_length = textstat.avg_sentence_length(article_text)
        fog_index = textstat.gunning_fog(article_text)
        word_count = len(article_text.split())
        complex_word_count = calculate_complex_word_count(article_text)
        avg_word_length = calculate_avg_word_length(article_text)
        syllables_per_word = textstat.syllable_count(article_text) / word_count if word_count > 0 else 0

        results.append({
            'URL_ID': url_id,
            'URL': url,
            'POSITIVE SCORE': positive_score,
            'NEGATIVE SCORE': negative_score,
            'POLARITY SCORE': polarity_score,
            'SUBJECTIVITY SCORE': subjectivity_score,
            'AVG SENTENCE LENGTH': avg_sentence_length,
            'FOG INDEX': fog_index,
            'WORD COUNT': word_count,
            'COMPLEX WORD COUNT': complex_word_count,
            'AVG WORD LENGTH': avg_word_length,
            'SYLLABLE PER WORD': syllables_per_word,
        })

    except Exception as e:
        print(f"Error processing URL_ID {url_id}: {e}")
        results.append({
            'URL_ID': url_id,
            'URL': url,
            'POSITIVE SCORE': 0,
            'NEGATIVE SCORE': 0,
            'POLARITY SCORE': 0,
            'SUBJECTIVITY SCORE': 0,
            'AVG SENTENCE LENGTH': 0,
            'FOG INDEX': 0,
            'WORD COUNT': 0,
            'COMPLEX WORD COUNT': 0,
            'AVG WORD LENGTH': 0,
            'SYLLABLE PER WORD': 0,
        })

output_df = pd.DataFrame(results)
final_output = pd.merge(output_template, output_df, on=['URL_ID', 'URL'], how='left')
final_output.to_csv(output_csv, index=False)

print("Processing complete. Results saved to output_results.csv")
