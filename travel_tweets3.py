import os
import json
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
class Tweet(object):
    def preprocess(self, tweet):
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
        text = tweet['text']
        text = text.replace("\\", "")
        tokens = tokenizer.tokenize(text)
        tokens = [token for token in tokens if self.is_ascii(token)]
        text = ' '.join(tokens)
        return text

    def html_unescape(self, text):
        from xml.sax.saxutils import unescape
        _html_unescape_table = {
            "&amp;": "&",
            "&quot;": '"',
            "&apos;": "'",
            "&gt;": ">",
            "&lt;": "<",
        }
        return unescape(text, _html_unescape_table)

    def is_ascii(self, token):
        return all([ord(c) < 128 and ord(c) >= 0 for c in token])

    def df(self, datapath):
        tdict, text, username, i = defaultdict(int), "", "", 1

        with open(datapath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    tweet = line
                    tweet_obj = json.loads(tweet)
                    username = tweet_obj['user']['screen_name']
                    text = ((self.preprocess(tweet_obj)) + " ")
                    if "#ttot" in text:
                        for _ in range(i):
                            tdict[i] = username, text
                            print(username, text)
                            i += 1
                        return text

s, tdict = Tweet(), {}
mypath = '/Users/priyanarayanasubramanian/Twitter Analysis/pac_nw_bb/'
for filename in os.listdir(mypath):
    if filename.endswith(".json"):
        print(os.path.join(mypath, filename))
        tdict = s.df(mypath+filename)

