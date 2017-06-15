import re
import json
from nltk.tokenize import TweetTokenizer

class Tweet(object):
    def preprocess(self, tweet):
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
        tdict = {}
        tdict[tweet['user']['screen_name']] =  tweet['text']
        text = tweet['text']
        username = [tweet['user']['screen_name']]
        #tweet_author = [tweet['user']['screen_name'] for tweet in tweets]
        #text = self.html_unescape(text)
        #text = re.sub(r"http\S+", "", text)
        # text = re.sub("@([a-z0-9_]+)", "RT @user1: who are @thing and @user2?", "")
        #text = re.sub(r"\@\w+", "", text)
        text = text.replace("\\", "")
        tokens = tokenizer.tokenize(text)
        #tokens = [token for token in tokens if len(token.replace('.', '')) > 1]
        # tokens = [token for token in tokens if token not in stopwords]
        tokens = [token for token in tokens if self.is_ascii(token)]
        text = ' '.join(tokens)
        # removed stopwords
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
        with open(datapath, 'r') as f:
            text, username = {}, ""
            for line in f:
                tweet = line
                tweet_obj = json.loads(tweet)
                username = tweet_obj['user']['screen_name']
                text[username] = ((self.preprocess(tweet_obj)) + " ")
            for name in text.keys():
                if "#ttot" in text[name]:
                    print(text[name])
                #print(username)
            #print(text)


            # text = ""
            # line = f.readline()  # read only the first tweet/line
            # tweet = json.loads(line)  # load it as Python dict
            # #print(json.dumps(tweet, indent=4))  # pretty-print
            # #tokens = preprocess(tweet['text'])
            # text += ((self.preprocess(tweet)) + " ")



s = Tweet()
s.df("last4.json")