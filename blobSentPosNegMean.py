from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time
import json
from DONTSHARE import *
# import sent_mod as s
import textblob
from textblob import TextBlob

class listener(StreamListener):

    tweet_out_txt = 'tweet_out.txt'
    tweet_out_csv = 'tweet_out.csv'

    rolling_neg = []
    rolling_pos = []
    rolling_sent = []
    mean_rolling_sent = 0.0
    mean_rolling_pos = 0.0
    mean_rolling_neg = 0.0
    
    def on_data(self, data):

        all_data = json.loads(data)
        tweet = all_data["text"]
        blob = TextBlob(tweet)
        polarity, subjectivity = blob.sentiment

        if polarity > 0.3:
            if len(listener.rolling_sent) >= 200:
                listener.rolling_sent.remove(listener.rolling_sent[0])
            if len(listener.rolling_pos) >= 200:
                listener.rolling_pos.remove(listener.rolling_pos[0])

            listener.rolling_sent.append(polarity)
            listener.rolling_pos.append(polarity)
            
            listener.mean_rolling_sent = sum(listener.rolling_sent) / len(listener.rolling_sent)
            listener.mean_rolling_pos = sum(listener.rolling_pos) / len(listener.rolling_pos)

            output = open(listener.tweet_out_txt, 'a')
            output.write(f'Tweet: {tweet}\nPolarity: {polarity}\nSubjectivity: {subjectivity}\nMean Rolling Sentiment: {listener.mean_rolling_sent}')
            output.close()

            tweet_time = time.time()
            csv_line = f'{tweet_time}, {listener.mean_rolling_sent}, {listener.mean_rolling_pos}, {listener.mean_rolling_neg}, {polarity}\n'
            with open(listener.tweet_out_csv, 'a') as fd:
                fd.write(csv_line)
                fd.close()

        elif polarity < -0.3:
            if len(listener.rolling_sent) >= 200:
                listener.rolling_sent.remove(listener.rolling_sent[0])
            if len(listener.rolling_neg) >= 200:
                listener.rolling_neg.remove(listener.rolling_neg[0])

            listener.rolling_sent.append(polarity)
            listener.rolling_neg.append(polarity)
            
            listener.mean_rolling_sent = sum(listener.rolling_sent) / len(listener.rolling_sent)
            listener.mean_rolling_neg = sum(listener.rolling_neg) / len(listener.rolling_neg)

            output = open(listener.tweet_out_txt, 'a')
            output.write(f'Tweet: {tweet}\nPolarity: {polarity}\nSubjectivity: {subjectivity}\nMean Rolling Sentiment: {listener.mean_rolling_sent}')
            output.close()

            tweet_time = time.time()
            csv_line = f'{tweet_time}, {listener.mean_rolling_sent}, {listener.mean_rolling_pos}, {listener.mean_rolling_neg}, {polarity}\n'
            with open(listener.tweet_out_csv, 'a') as fd:
                fd.write(csv_line)
                fd.close()

        print(f'Tweet: {tweet}\nPolarity: {polarity}\nSubjectivity: {subjectivity}\nMean Rolling Sentiment: {listener.mean_rolling_sent}')
           
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(APIKey, APISKey)
auth.set_access_token(AToken, ATokenSec)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Joe Biden, President Biden, JoeBiden"])
