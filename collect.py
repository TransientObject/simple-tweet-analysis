"""
Collects tweets by keyword and geographical bounding box
Uses Twitter Streaming APIs

Author: Solongo Munkhjargal
"""
# -*- coding: utf-8 -*-
import codecs
import os
import sys
import argparse
import logging
import json
import tweepy
import time
from datetime import datetime

class SListener(tweepy.StreamListener):
    """ Writes tweets """
    def __init__(self, output_file_prefix, current_time, datadir, api=None):
        self.api = api
        self.logger = logging
        self.datadir = datadir
        if not os.path.isdir(self.datadir):
            print "--- Created data directory {}---".format(self.datadir)
            os.mkdir(self.datadir)
        self.start_date = datetime.now().date()
        self.output_file_prefix = output_file_prefix
        filename = output_file_prefix+'.'+current_time+'.json'
        self.output = codecs.open(os.path.join(self.datadir, filename), \
                                  'wb+', "utf-8", "strict", 1)
    def on_status(self, status):
        ### write daily data
        json.dump(status._json, self.output)
        self.output.write('\n')
        current_date = datetime.now().date()
        ### if new day has started
        if (current_date - self.start_date).days == 1:
            self.output.close()
            self.start_date = current_date
            current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
            filename = self.output_file_prefix+'.'+current_time+'.json'
            self.output = codecs.open(os.path.join(self.datadir, filename)\
                                      , 'wb+', "utf-8", "strict", 1)
        elif (current_date - self.start_date).days != 0:
            self.logger.debug("File timestamping error: wrong day increment")
            sys.exit()
    def on_error(self, status_code):
        self.logger.debug("Received error code {}".format(status_code))
        # rate limit error
        if status_code == 420:
            ### disconnect the stream
            return False
    def on_limit(self, notice):
        self.logger.debug("Limit message : {}", notice)
        return
    def on_timeout(self):
        self.logger.debug("Timeout, sleeping for 60 seconds")
        time.sleep(60)
        return
def get_credentials(cfn):
    """ Load credentials file content"""
    with open(cfn, 'rb') as fin:
        credentials = json.loads(fin.read().strip())
    return credentials

def get_track_terms(track_fn):
    """Returns list of track terms"""
    with open(track_fn, 'rb') as fin:
        terms = fin.readlines()
    terms = filter(None, [term.strip().lower().translate(None, ' ') for term in terms])
    # remove duplicates
    terms = list(set(terms))
    return terms

def get_locations(loc_fn):
    """ Parses bounding box coordinates from file
        Returns string of lon,lat

        Bounding box coordinates format
            in the input file: SW_lat,SW_lon,NE_lat,NE_lon
        Output string format:
            SW_lon,SW_lat,NE_lon,NE_lat
        -----
        Had to transform the format because only Twitter uses this weird format
        Most other apis and apps use the input file format
    """
    with open(loc_fn, 'rb') as fin:
        coordinate_string = fin.readline()
    coordinates = coordinate_string.split(',')
    coordinates = [float(c.strip()) for c in coordinates]
    locs = [coordinates[1], coordinates[0], coordinates[3], coordinates[2]]
    return locs

def parse_cmd_args():
    """ Parses command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--credentials", help="path to credentials file")
    parser.add_argument("-t", "--track", help="File containing keyword terms to track")
    parser.add_argument("-op", "--outprefix", help="tweet/log file prefix")
    parser.add_argument("-d", "--datadir", help='directory to contain tweets')
    parser.add_argument("-g", "--geobox", help="File containing bounding box coordinates")
    parser.add_argument("-l", "--logdir", default="logs", help="Log file directory, default=logs")
    args = parser.parse_args()
    return args
def set_logger(args, current_time):
    """ Set up logger
    Args:
        args - ArgumentParser
        current_time - Time when the process started running
    Returns:
        None
    """
    # if log file directory does not exist, create one
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    # log basic info
    print "--- START TIME : {} ---\n".format(current_time)
    print "--- LOG FILE : {} ---\n".format(args.outprefix +'.'+ current_time + '.log')
    logging.basicConfig(
        filename=os.path.join(args.logdir, args.outprefix + '.' + current_time + '.log'),
        format='%(asctime)s %(message)s',
        datefmt='[%Y-%m-%d-%H-%M-%S]',
        level=logging.DEBUG)
    logging.debug("--- START TIME : %s ---", current_time)
    # log what arguments were passed
    print "--- ARGUMENTS ---"
    logging.debug("--- ARGUMENTS --- ")
    d_args = vars(args)
    for option in d_args.keys():
        print "--> {} : {}".format(option, d_args[option])
        logging.debug("--> %s : %s", option, d_args[option])

def authenticate(cred_fn):
    """ Performs authentication
        Args:
            cred_fn - path to credentials file
        Returns:
            auth - OAuthHandler
    """
    credentials = get_credentials(cred_fn)
    auth = tweepy.OAuthHandler(credentials['consumer_token'], credentials['consumer_secret'])
    auth.set_access_token(credentials['access_token'], credentials['access_token_secret'])
    return auth
def set_stream(args, auth, current_time):
    """
    Set up stream object
    Args:
        args - command line ArgumentParser
        auth - OAuthHandler object
        current_time - time at which the process started
    """
    api = tweepy.API(auth, wait_on_rate_limit=True,\
                           wait_on_rate_limit_notify=True)
    listener = SListener(args.outprefix, current_time, args.datadir, api=api)
    stream = tweepy.Stream(auth=auth, listener=listener)
    return stream

def main():
    """ The main functionality """
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    args = parse_cmd_args()
    track_terms, locs = None, None
    if args.track:
        if not args.track:
            raise ValueError('Keyword file is not supplied')
        track_terms = get_track_terms(args.track)
    if args.geobox:
        if not args.geobox:
            raise ValueError('Bounding box file is not supplied')
        locs = get_locations(args.geobox)
    set_logger(args, current_time)
    auth = authenticate(args.credentials)
    print "--- Authentication complete ---"
    logging.debug("--- Authentication complete---")
    logging.debug("--- Starting stream ---")
    stream = set_stream(args, auth, current_time)
    if args.track:
        print "--- Track terms ---\n"
        print "--> The number of track terms : {} <--".format(len(track_terms))
        print track_terms
        logging.debug("--- Track terms ---\n"  + ' '.join(track_terms))
        logging.debug("--> The number of track terms : %d <--", len(track_terms))
        logging.debug("--- Started tracking ---")
        """
        track - comma separated list of keywords
                commas = OR
                spaces = AND
        matches with - tweet text, expanded and display urls,
                       hashtag text, and screen_name
        maxLength of one keyword - 1 to 60chars (bytes)
        Twitter does not trim punctuation and special characters in the kw
            But, kw "hello" with match with tweet "hello."
        kw containing punctuation does not match with hashtag and mentions
        UTF8 chars match exactly
        """
        stream.filter(track=track_terms)
    elif args.geobox:
        """
        Does not check user location field
        """
        print "--- Bounding box : {}---\n".format(str(locs))
        logging.debug("--- Bounding box : %s---", str(locs))
        stream.filter(locations=locs)
    else:
        print "--- Listening to the Public Stream ---"
        logging.debug('--- Listening to the Public Stream---')
        stream.sample()
if __name__ == '__main__':
    main()
