import os
import re
import string
import sys

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pycountry
import tweepy
from langdetect import detect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import STOPWORDS, WordCloud

nltk.download("vader_lexicon")

# Authentication
consumer_key = os.environ["TWITTER_CONSUMER_KEY"]
consumer_secret = os.environ["TWITTER_CONSUMER_SECRET"]
access_token = os.environ["TWITTER_ACCESS_TOKEN"]
access_token_secret = os.environ["TWITTER_TOKEN_SECRET"]


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Sentiment Analysis
def percentage(part, whole):
    return 100 * float(part) / float(whole)


keyword = input("Please enter keyword or hashtag to search: ")
noOfTweet = int(input("Please enter how many tweets to analyze: "))
tweets = tweepy.Cursor(api.search, q=keyword).items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []
for tweet in tweets:

    print(tweet.text)
    tweet_list.append(tweet.text)
    analysis = TextBlob(tweet.text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    neg = score["neg"]
    neu = score["neu"]
    pos = score["pos"]
    comp = score["compound"]
    polarity += analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(tweet.text)
        negative += 1
    elif pos > neg:
        positive_list.append(tweet.text)
        positive += 1

    elif pos == neg:
        neutral_list.append(tweet.text)
        neutral += 1

positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, ".1f")
negative = format(negative, ".1f")
neutral = format(neutral, ".1f")

# Number of Tweets (Total, Positive, Negative, Neutral)
tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
print("total number: ", len(tweet_list))
print("positive number: ", len(positive_list))
print("negative number: ", len(negative_list))
print("neutral number: ", len(neutral_list))

# Creating PieCart
labels = [
    "Positive [" + str(positive) + "%]",
    "Neutral [" + str(neutral) + "%]",
    "Negative [" + str(negative) + "%]",
]
sizes = [positive, neutral, negative]
colors = ["yellowgreen", "blue", "red"]
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.style.use("default")
plt.legend(labels)
plt.title("Sentiment Analysis Result for keyword= " + keyword + "")
plt.axis("equal")
plt.show()

tweet_list.drop_duplicates(inplace=True)

# Cleaning Text (RT, Punctuation etc)
# Creating new dataframe and new features
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]
# Removing RT, Punctuation etc
remove_rt = lambda x: re.sub("RT @\w+: ", " ", x)
rt = lambda x: re.sub("(@[A-Za-z0–9]+)|([0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()
tw_list.head(10)

# Calculating Negative, Positive, Neutral and Compound values
tw_list[["polarity", "subjectivity"]] = tw_list["text"].apply(
    lambda Text: pd.Series(TextBlob(Text).sentiment)
)
for index, row in tw_list["text"].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score["neg"]
    neu = score["neu"]
    pos = score["pos"]
    comp = score["compound"]
    if neg > pos:
        tw_list.loc[index, "sentiment"] = "negative"
    elif pos > neg:
        tw_list.loc[index, "sentiment"] = "positive"
    else:
        tw_list.loc[index, "sentiment"] = "neutral"
        tw_list.loc[index, "neg"] = neg
        tw_list.loc[index, "neu"] = neu
        tw_list.loc[index, "pos"] = pos
        tw_list.loc[index, "compound"] = comp

tw_list.head(10)


# Creating new data frames for all sentiments (positive, negative and neutral)
tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]


def count_values_in_column(data, feature):
    total = data.loc[:, feature].value_counts(dropna=False)
    percentage = round(
        data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2
    )
    return pd.concat([total, percentage], axis=1, keys=["Total", "Percentage"])


# Count_values for sentiment
count_values_in_column(tw_list, "sentiment")

# create data for Pie Chart
pichart = count_values_in_column(tw_list, "sentiment")
names = pichart.index
size = pichart["Percentage"]

# Create a circle for the center of the plot
my_circle = plt.Circle((0, 0), 0.7, color="white")
plt.pie(size, labels=names, colors=["green", "blue", "red"])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# Function to Create Wordcloud
def create_wordcloud(text):
    mask = np.array(Image.open("cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(
        background_color="white",
        mask=mask,
        max_words=3000,
        stopwords=stopwords,
        repeat=True,
    )
    wc.generate(str(text))
    wc.to_file("wc.png")
    print("Word Cloud Saved Successfully")
    path = "wc.png"
    # display(Image.open(path))
