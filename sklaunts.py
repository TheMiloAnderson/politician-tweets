from utils import parse_sentence
from utils import classify_model
from utils import classify_sentence
from utils import setup_database
from utils import add_to_database
from poli_word_count import class_maker
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from token_maker import get_string

tweets_df = pd.read_csv('words/ExtractedTweets.csv')

setup_database()

for ti in range(len(tweets_df)):
    class_maker(ti)
