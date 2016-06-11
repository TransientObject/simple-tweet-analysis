#!/usr/bin/env python
"""
Collects friends' userids

- python get_friends.py output_directory.path list_of_user_ids.file 
"""
import time
import pandas as pd
import Queue
import tweepy
import csv
import json
import sys
import os
from tweepy.parsers import JSONParser
from os.path import isfile
import time

def main(credentials='rest_credentials.json'):
    with open(credentials) as cin:
        c = json.load(cin)
    auth2 = tweepy.AppAuthHandler(c['CONSUMER_KEY'] , c['CONSUMER_SECRET'])
    api = tweepy.API(auth2 , wait_on_rate_limit =True, wait_on_rate_limit_notify=True, timeout=10000, retry_delay=60)

    if (not api):
        print ("Could not Authenticate")
        sys.exit(-1)

    print "--- Authentication Successful ---"

    outdir = sys.argv[1]
    idfile = sys.argv[2]
    ls = open(idfile).read().splitlines()

    for uid in ls:
        print uid
        friends = []
        try:
            for page in tweepy.Cursor(api.friends_ids, user_id=uid).pages():
                friends.extend(page)

	    friends = [str(t) for t in friends]

            with open(os.path.join(outdir , uid) , 'wb') as fout:
			
                fout.write(",".join(friends) + os.linesep)

        except Exception , e :
            print str(e)

if __name__ == "__main__":
    main()
