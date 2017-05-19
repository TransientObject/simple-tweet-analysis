#!/usr/bin/env python
import queue
import tweepy
import csv
import json
import sys
import os
from os.path import isfile
import time
import traceback

def get_uniq_uid(infile , outfile = "uid.csv"):
    _s = time.time()
    uid = dict()
    foutfile = open(outfile , "w+")    
    fout = csv.writer(foutfile, delimiter = ",")
    with open(infile) as fin:
        for line in fin:
            j = json.loads(line)
            i = j['user']['id']
            if i not in uid:
                uid[i] = 1
            else:
                uid[i] += 1
    for k in uid:
        fout.writerow([ k , uid[k]])
    foutfile.close()
    print("Completed in : ", (time.time() - _s) , "seconds")
def track_u(user_id, api , outdir = "output/"):
    '''
    user_id - string 
    api- tweepy.API object 
    outdir - directory to store user json files 
    '''
    sinceId = None
    maxId = -1
    print("--- Started tracking user : " , user_id)
    fname = outdir + user_id + ".json"
    if isfile(fname):
        fname = outdir + "new." + user_id + ".json" 
    
    with open(fname , "a+") as fout:
        while True:
            try:
                if maxId <= 0:
                    if (not sinceId):
                        new_tweets = api.user_timeline(user_id = user_id , count = 200 , include_rts=True)
                    else:
                        new_tweets = api.user_timeline(user_id = user_id , count = 200 , since_id=sinceId, include_rts=True)
                else:
                    if (not sinceId):
                        new_tweets = api.user_timeline(user_id = user_id , count = 200 , max_id = str(maxId - 1), include_rts=True)
                    else:
                        new_tweets = api.user_timeline(user_id = user_id , count = 200 , since_id=sinceId , max_id = str(maxId - 1), include_rts=True)
                if (not new_tweets):
                    fout.close()
                    return
                for tweet in new_tweets:
                    json.dump(tweet._json , fout)
                    fout.write(os.linesep) 
                tweetCount += len(new_tweets)
                maxId = new_tweets[-1].id
                #sinceId = new_tweets[0].id
            except tweepy.TweepError as e:
                print(("--- ERROR --- " + str(e)))
                break                

def track_all_u(allu_f="uid.csv", tu_f="uid_from_target.txt", credentials = "rest_credentials.json" , puf = "processed_users.file" ):
    '''
    allu_f - all user file 
    tu_f - target user file
    puf - processed users file
    '''
    _start  = time.time()
    print("--- Starting @ %s ---"%(time.strftime("%a, %d %b %Y %H:%M:%S",
                                               time.localtime())))
    with open(puf) as f:
        pu = f.read().splitlines()

    pu = [i.split('.')[0] for i in pu]

    print("--- Already collected : " , len(pu))    
    af = open(allu_f)
    tf = open(tu_f)
    afin = csv.reader(af , delimiter = ",")
    tfin = csv.reader(tf , delimiter = ",")
    pq = queue.PriorityQueue()
    # get users from target cities
    tuset = set()
    totalu = 0
    for line in tfin:
        totalu += 1
        tuset.add(line[0])
    print("--- # user from target: " , totalu)
    print("--- # of Unique Users : " , len(tuset))
    # populate priority queue
    try:
            for line in afin:
                n = int(line[1]) if line[0] in tuset else int(line[1]) + 1
                if line[0] not in pu:
                    pq.put(( n , line[0]))
            print("--- Size of Priority Queue : " , pq.qsize())
    except Exception as e:
        print((traceback.format_exc()))

    # Do the authentication
    with open(credentials) as cin:
        c = json.load(cin)        
    auth2 = tweepy.AppAuthHandler(c['CONSUMER_KEY'] , c['CONSUMER_SECRET'])
    api = tweepy.API(auth2 , wait_on_rate_limit =True, wait_on_rate_limit_notify=True)
    if (not api):
        print ("Could not Authenticate")
        sys.exit(-1)
    print("--- Authentication Successful")
    # Now start tracking users
    while not pq.empty():
        user = pq.get()[1]
        track_u(user , api)
    print("--- Total time spent : " , (time.time()  - _start))
    af.close()
    tf.close()
    
if __name__ == "__main__":
    track_all_u()
