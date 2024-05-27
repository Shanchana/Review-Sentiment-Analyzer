import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Define the function to preprocess text
def preprocess_text(text_data):
    preprocessed_text = []
    stop_words = set(stopwords.words('english'))
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_sentence = ' '.join(
            token.lower() for token in word_tokenize(sentence) if token.lower() not in stop_words
        )
        preprocessed_text.append(preprocessed_sentence)
    return preprocessed_text

# Load and preprocess training data
def load_data():
  data = pd.read_csv('C:\\Users\\GUNA\\Desktop\\python\\flipkart.csv')
  data['review'] = preprocess_text(data['review'].values)
  return data

# Encoding as 0 and 1's
def encoding(data):
  pos_neg = []
  for i in range(len(data['rating'])):
    if data['rating'][i] >= 5:
      pos_neg.append(1)
    else:
      pos_neg.append(0)

  data['label'] = pos_neg
  return data

# Vectorize the reviews
def training(data):
  cv = TfidfVectorizer(max_features=2500)
  X = cv.fit_transform(data['review']).toarray()
  y = data['label']
  #split data
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
  #decision tree classifier
  model = DecisionTreeClassifier(random_state=0)
  model.fit(x_train, y_train)
  return cv, model



# Define the function to scrape product reviews from Amazon
def get_soup(url, headers):
    page = requests.get(url, headers=headers)
    if page.status_code != 200:
        print(f"Failed to retrieve page: {url}")
        return None
    return BeautifulSoup(page.content, "html.parser")

def extract_reviews(soup):
    reviews = []
    review_blocks = soup.find_all("div", {"data-hook": "review"})
    for block in review_blocks:
        review_text = block.find("span", {"data-hook": "review-body"})
        if review_text:
            reviews.append(review_text.get_text(strip=True))
    return reviews


def scraping(url, headers):
    soup = get_soup(url, headers)
    if not soup:
        return [], "default_title", "No rating available"

    title = soup.find("span", {"id": "productTitle"})
    title = title.get_text(strip=True) if title else "default_title"
    #print("Product Title:", title)

    rating = soup.find("span", {"class": "a-icon-alt"})
    rating = rating.get_text(strip=True).split(" ")[0] if rating else "No rating available"
    #print("Rating:", rating)

    all_reviews = extract_reviews(soup)
    if not all_reviews:
        print("No reviews found on the main page. Checking paginated review pages.")
        page_number = 1
        while True:
            next_page_url = f"https://www.amazon.in/product-reviews/{url.split('/')[-1]}?pageNumber={page_number}"
            soup = get_soup(next_page_url, headers)
            if not soup:
                break
            reviews = extract_reviews(soup)
            if not reviews:
                break
            all_reviews.extend(reviews)
            page_number += 1
    #print("Reviews:")
    #for review in all_reviews:
        #print(review)
    return all_reviews, title, rating




# Convert reviews to a DataFrame for further analysis
def analyze(all_reviews,cv,model):
  review_data = pd.DataFrame(all_reviews, columns=['review'])
  # Preprocess the reviews
  review_data['review'] = preprocess_text(review_data['review'].values)
  # Vectorize the reviews
  X_new = cv.transform(review_data['review']).toarray()
  # Predict using the trained model
  review_data['label'] = model.predict(X_new)
  # Perform sentiment analysis
  sentiment = SentimentIntensityAnalyzer()
  review_data['Positive'] = review_data['review'].apply(lambda x: sentiment.polarity_scores(x)["pos"])
  review_data['Negative'] = review_data['review'].apply(lambda x: sentiment.polarity_scores(x)["neg"])
  review_data['Neutral'] = review_data['review'].apply(lambda x: sentiment.polarity_scores(x)["neu"])

  return review_data

# Summarize the sentiment scores
def summarize(review_data):
  x = review_data['Positive'].sum()
  y = review_data['Negative'].sum()
  z = review_data['Neutral'].sum()
  
  total_sum = x + y + z
  Positive_p = (x / total_sum) * 100
  Negative_p = (y / total_sum) * 100
  Neutral_p = (z / total_sum) * 100
  
  a = int(Positive_p)
  b =int(Negative_p)
  c= int(Neutral_p)
  return a,b,c

def sentiment_score(a, b, c):
    if a > b and a > c:
        return "Positive"
    elif b > a and b > c:
        return "Negative"
    else:
        return "Neutral"

def main_process(url):
    # Load and preprocess training data
    data = load_data()
    data = encoding(data)
    
    # Train the model
    cv, model = training(data)
    
    # Scrape Amazon reviews
    #url = "https://amzn.in/d/9u1eCqQ"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'}
    all_reviews, title, rating = scraping(url, headers)
    
    if not all_reviews:
        print("No reviews found.")
        return
    
    # Analyze the scraped reviews
    review_data = analyze(all_reviews, cv, model)
    
    # Summarize sentiment scores
    Positive_p, Negative_p, Neutral_p = summarize(review_data)
    
    # Display overall sentiment
    overallresult =sentiment_score(Positive_p, Negative_p, Neutral_p)
    return {
       "title" : title,
       "rating" : rating,
       "positive" : Positive_p,
       "negative" : Negative_p,
       "neutral" : Neutral_p,
       "overall_r" : overallresult
    }




