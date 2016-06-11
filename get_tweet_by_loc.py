"""
REST API
Collects tweets sent from within a bounding box
"""

#!/usr/bin/env python
import sys
from tweepy import TweepError
import time
import tweepy
import json
import os
def get_locations(loc_fn):
    """Returns list of lat,lon
    """
    with open(loc_fn, 'rb') as fin:
        coordinate_string = fin.readline()
    coordinates = coordinate_string.split(',')
    coordinates = [float(c.strip()) for c in coordinates]
    return coordinates

def by_loc(credentials, loc_fn):
    with open(credentials) as cin:
        c = json.load(cin)
    auth2 = tweepy.AppAuthHandler(c['consumer_token'], c['consumer_secret'])
    api = tweepy.API(auth2, wait_on_rate_limit=True, wait_on_rate_limit_notify=True,\
                     timeout=10000, retry_delay=60)

    if not api:
        print "Could not Authenticate"
        sys.exit(-1)

    print "--- Authentication Successful ---"
    l = get_loc(loc_fn)
    tout = open(os.path.join(loc_fn+"loc-kw-03-31-04-01."+time.strftime('%Y%m%d-%H%M%S')+ '.json'), 'a+')
    i = 0
    while True:
      try:
        for tweet in tweepy.Cursor(api.search, geocode=l, rpp=100, lang='en').items():
            json.dump(tweet._json, tout)
            tout.write('\n')
            print "Written {} tweets.".format(i)
            i += 1
      except TweepError as e:   
        print str(e)

    tout.close()

def get_loc(loc_fn, radius=100):
    loc = get_locations(loc_fn)	
    print loc 
    x = (loc[0], loc[1])
    y = (loc[2], loc[3])
    center = (loc[0] + loc[2])/2, (loc[1] + loc[3])/2
    r = ','.join(map(str,[center[0], center[1], radius]))+"km"
    print r 
    return r 

if __name__ == "__main__":
    by_loc(sys.argv[1], sys.argv[2])
