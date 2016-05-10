#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from collections import Counter
import operator
import cartopy
import shapely
import cartopy.io.shapereader as shpreader
import traceback
import re 
import argparse
import csv
import time
import json
import sys
import pandas as pd
import multiprocessing as mp
import string
import os
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
import cPickle

def load_offsets(datapath, pickle=True):
    offsets = []
    offset = 0
    with open(datapath, 'rb') as fin:
        for line in fin:
            offsets.append(offset)
            offset += len(line)
    if pickle:
        with open(datapath + 'line-offset.pkl', 'wb') as fout:
            cPickle.dump(offsets, fout)

    return offsets
def has_kw(text, kwlist):
    """
    Return matches would return matching keywords
    
    """
    # if the keyword list is empty return True
    if kwlist is None:
        return True
    
    words = text.split()
    unigram_track = set([i for i in kwlist if ' ' not in i ])
    ngram_track = [' ' + i + ' ' for i in kwlist if ' ' in i]
    matches = filter(lambda x: x in unigram_track, words) 
    for track in ngram_track:
        if track in text:
            matches.extend([track.strip()]*text.count(track))
        if text.startswith(track.strip()):
            matches.extend([track.strip()])
        if  text.endswith(track.strip()):
            matches.extend([track.strip()])
    return matches

def find_ngrams(text, n):
    input_list = text.split()
    ret = zip(*[input_list[i:] for i in range(n)])
    return [' '.join(i) for i in ret]

def html_unescape(text):
    from xml.sax.saxutils import unescape
    _html_unescape_table = {
        "&amp;":"&",
        "&quot;":'"',
        "&apos;":"'",
        "&gt;":">",
        "&lt;":"<",
        }
    return unescape(text, _html_unescape_table)

def isurl(token):
    return token[:4] == 'http' or token[:5]=='https'

def is_ascii(token):
    return all([ord(c) < 128 and ord(c) >=0 for c in token])

def get_track_terms(track_fn):
    """Returns list of track terms in the file"""
    with open(track_fn, 'rb') as fin:
        terms = fin.readlines()
    terms = [term.strip().lower() for term in terms]
    terms = list(set(terms))
    return terms

def is_retweet(x, include_old=True):
      if x.get('retweeted_status', None):
          return True
      else:
          if include_old:
              if x['text'][:4] == 'rt @':
                  return True
          return False

def preprocess(tweet, stopwords):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)    
    text=tweet['text']
    text = html_unescape(text)
    text = text.replace("\\", "")
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if len(token.replace('.','')) > 1]
    tokens = [token for token in tokens if token not in stopwords]
    ### we need to remove ascii tokens to make bigram search easy
    tokens = [token for token in tokens if is_ascii(token)] 
    text =' '.join(tokens)
    # removed stopwords
    return text
def sentiment_preprocess(tweet, stopwords):
    import string
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    text=tweet['text']
    text = html_unescape(text)
    text = text.replace("\\", "")
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if len(token.replace('.','')) > 1]
    tokens = [token for token in tokens if token not in stopwords]
    ### remove ascii tokens to make bigram search easy
    tokens = [token for token in tokens if is_ascii(token)]
    tokens = [token.lstrip(string.punctuation) for token in tokens]
    text =' '.join(tokens)
    # removed stopwords
    return text
def load_states_shapefile(shpfile="shapes/2015/tl_2015_us_state.shp"):
    d = {}
    for tmp in shpreader.Reader(shpfile).records():
        d[tmp.attributes['NAME'].lower()] = tmp.geometry
    return d
def geofilter(tweet_obj, shp):
    # modified resolve_state
    if tweet_obj.get('coordinates', None) and tweet_obj['coordinates'].get('coordinates', None):
        if shp is None:
            return "geotagged"
        point = shapely.geometry.Point(tuple(tweet_obj['coordinates']['coordinates'])) 
        for k,v in shp.items():
            if v.contains(point):
                return k
        else:
            return None
    elif tweet_obj.get('place', None):
        if shp is None:
            return "geotagged"
        # bounding box (minx, miny, maxx, maxy)
        for k,v in shp.items():
            try:
                c = tweet_obj['place']['bounding_box']['coordinates'][0]
            except Exception ,e:
                return None
            bb  = shapely.geometry.Polygon(c)
            (minx, miny, maxx, maxy) = bb.bounds
            if v.contains(shapely.geometry.Point((minx, miny))) and \
               v.contains(shapely.geometry.Point((maxx, maxy))) and \
               v.contains(shapely.geometry.Point((minx, maxy))) and \
               v.contains(shapely.geometry.Point((maxx, miny))): 
                    return k
        else:
            return None
    else:
        return None
def get_sentiment(text):
    """Returns valence score of text, uses LabMT, removes stopValues"""
    from labMTsimple import storyLab
    lang = 'english'
    labMT,labMTvector,labMTwordList = storyLab.emotionFileReader(stopval=0.0,lang=lang,returnVector=True)
    textValence,textFvec = storyLab.emotion(text,labMT,shift=True,happsList=labMTvector)
    textStoppedVec = storyLab.stopper(textFvec,labMTvector,labMTwordList,stopVal=1.0)
    textValence = storyLab.emotionV(textStoppedVec,labMTvector)
    return textValence

def load_us_shape():
    us_shp = []
    shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()
    for country in countries:
        if country.attributes['adm0_a3'] == 'USA':
            us_shp.append(country)
    us_shp = us_shp[0].geometry
    return {"us":us_shp}
def has_media(tweet_obj, instagram=False, photo_only=False):
    """Checks whether tweet contains a media entity"""
    if tweet_obj.get('entities', None) is None:
        return 0
    else:
        if tweet_obj['entities'].get('media', None) is None and instagram == False:
            return 0
        else:
            for entity in tweet_obj['entities'].get('media', []):
                if photo_only:
                    if entity['type'] == 'photo':
                        return 1
                    else:
                        continue
                else:
                    return 1
            if instagram:
                if tweet_obj['entities'].get('urls', None):
                    for url in tweet_obj['entities']['urls']:
                        if 'instagram' in url['expanded_url']:
                            return 1
                        else:
                            continue
                    return 0
                else:
                    return 0
            else:
                return 0
class Workers(object):
    def __init__(self, subroutine, datapath, output, kw, 
                    line_offsets, nor, oldr, english_only,
                    geofilter, shapefile, photo_only, instagram,
                    topk, ngram): 
    
        self.numprocs = mp.cpu_count()
        self.outfile = output
        self.datapath = datapath
        self.noretweet = nor
        self.no_oldr = oldr
        self.english_only = english_only
        self.photo_only = photo_only
        self.instagram = instagram
        self.topk, self.ngram = int(topk), int(ngram)

        self.geofilter = geofilter

        if self.geofilter:
            self.geofilter = geofilter.strip().lower()
        
        if self.geofilter == 'us':
            self.shape = load_us_shape()
        elif self.geofilter == 'states':
            self.shape = load_states_shapefile()
        else:
            # custom geofilter is not yet supported 
            self.shape = None

        # total number of documents and terms for normalization
        self.total_tf = 0
        self.total_df = 0

        with open('rainbow.txt', 'rb') as fin:
            self.stopword = fin.readlines()
            self.stopword = [word.strip().lower() for word in self.stopword]
            self.stopword = set(self.stopword)
        
        if kw is None:
            raise ValueError('a Keyword list must be specified')

        self.track = get_track_terms(kw)
         
        if line_offsets:
            with open(line_offsets, 'rb') as fin:
                self.inlist = cPickle.load(fin)
        else:
            self.inlist = load_offsets(datapath, pickle=True) 
        print '--- # Tweets to preprocess :',len(self.inlist)
        print '--- Subroutine name : ' , subroutine
        print '--- Output will be stored in : ' , self.outfile
    
        self.inq = mp.Queue()
        self.outq = mp.Queue()
        self.pin = mp.Process(target=self.load_input_q , args = ())
        if subroutine == 'tf':
            self.ps = [mp.Process(target = self.tf, args=()) for x in range(self.numprocs)]
            self.pout = mp.Process(target=self.read_output_q_json, args = ())
        elif subroutine == 'df':
            self.ps = [mp.Process(target = self.df, args=()) for x in range(self.numprocs)]
            self.pout = mp.Process(target=self.read_output_q_json, args = ())
        elif subroutine == 'media':
            self.ps = [mp.Process(target = self.media, args=()) for x in range(self.numprocs)]
            self.pout = mp.Process(target=self.read_output_q_media, args = ())
        elif subroutine == 'cr':
            self.ps = [mp.Process(target = self.count_retweet, args=()) for x in range(self.numprocs)]
            self.pout = mp.Process(target=self.read_output_q_cnt, args = ()) 
        elif subroutine == 's':
            self.ps = [mp.Process(target = self.sentiment, args=()) for x in range(self.numprocs)]
            self.pout = mp.Process(target=self.read_output_q_sentiment, args = ())
        elif subroutine =='tc':
            self.ps = [mp.Process(target = self.term_correlation, args=()) for x in range(self.numprocs)]
            self.pout = mp.Process(target=self.read_output_q_tc, args = ())
        elif subroutine =='mc':
            self.ps = [mp.Process(target = self.most_common, args=()) for x in range(self.numprocs)]
            self.pout = mp.Process(target=self.read_output_q_mc, args = ()) 
        elif subroutine == 'gp':
            self.ps = [mp.Process(target = self.profiler, args=()) for x in range(self.numprocs)]
            self.pout = mp.Process(target=self.read_output_q_profiler, args=())
        else:
            print "Invalid subroutine name"
            sys.exit()
    def start(self):
                
        self.pin.start()
        self.pout.start()
        for p in self.ps:
            p.start()

        self.pin.join()

        i = 0

        for p in self.ps:
            p.join()
            print "Done" , i
            i += 1
        self.pout.join()
    def load_input_q(self):
        for offset in self.inlist:
            self.inq.put(offset)

        for i in range(self.numprocs):
            self.inq.put("STOP")
    def read_output_q_json(self):
        cur = 0
        stop = 0
        #result = defaultdict(int)
        offsets = set([])
        per_place = defaultdict(dict)

        with open(self.outfile , "wb") as outfile:
            for works in range(self.numprocs):
                for offset, val, tf, geotag in iter(self.outq.get , "STOP"):
                    if offset in offsets:
                        # skip if offset has been processed
                        continue
                    else:
                        offsets.add(offset)
                        geotag = str(geotag)
                        for kw in val:
                            per_place[geotag][kw] = per_place[geotag].get(kw, 0) + 1
    
                        per_place[geotag]['total_tf'] = per_place[geotag].get('total_tf', 0) + tf
                        per_place[geotag]['total_df'] = per_place[geotag].get('total_df', 0) + 1
    
                        self.total_tf += tf
                        self.total_df += 1
            json.dump(per_place, outfile)
            outfile.write("\n")
    def read_output_q_media(self):
        cur = 0
        stop = 0
        result = defaultdict(dict)
        offsets = set([])
        per_place = defaultdict(dict)
        with open(self.outfile , "wb") as outfile:
            for works in range(self.numprocs):
                for offset, val, geotag, media in iter(self.outq.get , "STOP"):
                    if offset in offsets:
                        # skip if offset has been processed
                        continue
                    else:
                        offsets.add(offset)
                        geotag = str(geotag)
                        for kw in val:
                            dd = per_place[geotag].get(kw, {})
                            dd['media'] = dd.get('media', 0)  + media
                            dd['total_df'] = dd.get('total_df', 0) + 1
                            per_place[geotag][kw] = dd
                
            for g in per_place:
                for k in per_place[g]:
                    per_place[g][k]['fraction'] = per_place[g][k]['media']*1.0/per_place[g][k]['total_df']
            json.dump(per_place, outfile)
            outfile.write('\n')
    def read_output_q_profiler(self):
        cur = 0
        stop = 0
        result = Counter(dict())
        offsets = set([])
        with open(self.outfile , "wb") as outfile:
            for works in range(self.numprocs):
                for offset, counter in iter(self.outq.get , "STOP"):
                    if offset in offsets:
                        # skip if offset has been processed
                        continue
                    else:
                        offsets.add(offset)
                        result =result + counter
            json.dump(result, outfile)
            outfile.write('\n')
    def read_output_q_tc(self):
        cur = 0
        stop = 0
        result = defaultdict(dict)
        offsets = set([])
        per_place = defaultdict(dict)
        with open(self.outfile , "wb") as outfile:
            for works in range(self.numprocs):
                for offset, val, ngrams, geotag in iter(self.outq.get , "STOP"):
                    if offset in offsets:
                        # skip if offset has been processed
                        continue
                    else:
                        offsets.add(offset)
                        geotag = str(geotag)
                        for kw in val:
                            dd = per_place[geotag].get(kw, {})
                            for ngram in ngrams:
                                if ngram == kw:
                                    continue
                                dd[ngram] = dd.get(ngram, 0)  + 1

                            dd['total_df'] = dd.get('total_df', 0) + 1
                            per_place[geotag][kw] = dd
            for g in per_place:
                for k in per_place[g]:
                    sorted_d = sorted(per_place[g][k].items(), key=operator.itemgetter(1), reverse=True)
                    values = set(sorted(list(set([i[1] for i in sorted_d])), reverse=True)[:self.topk+1])
                    sorted_d = filter(lambda x: x[1] in values, sorted_d) 
                    per_place[g][k] = dict(sorted_d)
            json.dump(per_place, outfile)
            outfile.write('\n')
    def read_output_q_mc(self):
        cur = 0
        stop = 0
        result = defaultdict(dict)
        offsets = set([])
        per_place = defaultdict(dict)
        with open(self.outfile , "wb") as outfile:
            for works in range(self.numprocs):
                for offset, ngrams, geotag in iter(self.outq.get , "STOP"):
                    if offset in offsets:
                        # skip if offset has been processed
                        continue
                    else:
                        offsets.add(offset)
                        geotag = str(geotag)
                        for kw in ngrams:
                            per_place[geotag][kw] = per_place[geotag].get(kw, 0) + 1
                        per_place[geotag]['total_df'] = per_place[geotag].get('total_df', 0) + 1
            for g in per_place:
                sorted_d = sorted(per_place[g].items(), key=operator.itemgetter(1), reverse=True)
                values = set(sorted(list(set([i[1] for i in sorted_d])), reverse=True)[:self.topk+1])
                sorted_d = filter(lambda x: x[1] in values, sorted_d)
                per_place[g] = dict(sorted_d)
            json.dump(per_place, outfile)
            outfile.write('\n')
    def read_output_q_sentiment(self):
        cur = 0
        stop = 0
        result = defaultdict(dict)
        offsets = set([])
        per_place = defaultdict(dict)
        with open(self.outfile , "wb") as outfile:
            for works in range(self.numprocs):
                for offset, val, geotag, text in iter(self.outq.get , "STOP"):
                    if offset in offsets:
                        # skip if offset has been processed
                        continue
                    else:
                        offsets.add(offset)
                        geotag = str(geotag)
                        for kw in val:
                            dd = per_place[geotag].get(kw, {})
                            #dd['text'] = dd.get('text', "")  +" "+ text
                            dd['text'] = dd.get('text',[])
                            dd['text'].append(text)
                            dd['total_df'] = dd.get('total_df', 0) + 1
                            per_place[geotag][kw] = dd
            for g in per_place:
                for enum, k in enumerate(per_place[g]):
                    per_place[g][k]['magnitude'] = get_sentiment(' '.join(per_place[g][k]['text']))
                    #### delete the following line 
                    per_place[g][k]['text'] = None
                    print "----{}----{} computed".format(enum, k)
            json.dump(per_place, outfile)
            outfile.write('\n')

    def read_output_q_cnt(self):
        cur = 0
        stop = 0
        offsets = set([])
        per_place = defaultdict(dict)
        with open(self.outfile , "wb") as outfile:
            for works in range(self.numprocs):
                for offset, val, geotag, cnt in iter(self.outq.get , "STOP"):
                    if offset in offsets:
                        # skip if offset has been processed
                        continue
                    else:
                        offsets.add(offset)
                        geotag = str(geotag)
                        for kw in val:
                            dd = per_place[geotag].get(kw, {})
                            dd['rt_cnt'] = dd.get('rt_cnt', 0)  + cnt
                            dd['rt_flag'] = dd.get('rt_flag', 0) + (cnt > 0)
                            dd['total_df'] = dd.get('total_df', 0) + 1
                            per_place[geotag][kw] = dd
            for g in per_place:
                for k in per_place[g]:
                    per_place[g][k]['average'] = per_place[g][k]['rt_cnt']*1.0/per_place[g][k]['total_df']
                    per_place[g][k]['fraction'] = per_place[g][k]['rt_flag']*1.0/per_place[g][k]['total_df']
            json.dump(per_place, outfile)
            outfile.write('\n')

    def read_output_q(self):
        cur = 0
        stop = 0
        buffer = {}
        with open(self.outfile , "wb") as outfile:
            for works in range(self.numprocs):
                for val in iter(self.outq.get , "STOP"):
                    outfile.write(val  + "\n")
    def tf(self):
        with open(self.datapath, 'rb') as fin:
            for offset in iter(self.inq.get , "STOP"):
                try:
                    fin.seek(offset)
                    tweet = fin.readline()
                    tweet_obj = json.loads(tweet)
                    if self.noretweet and is_retweet(tweet_obj, include_old=(not self.no_oldr)):
                        continue
                    if self.english_only and tweet_obj.get("lang", None) != 'en':
                        continue
                    # geofilter 
                    geotag = None
                    if self.geofilter:
                        geotag = geofilter(tweet_obj, self.shape)
                        if geotag == None:
                            continue
                    text = preprocess(tweet_obj, self.stopword)

                    kws = has_kw(text, self.track)

                    ## now counts the ngram terms as one term
                    total_tf = len(kws) + (len(text.split()) - sum(map(lambda x: len(x.split()), kws)))

                    self.outq.put((offset, kws, total_tf, geotag))
                except ValueError, v:   
                    continue
                except Exception as e:
                    print traceback.format_exc()
        self.outq.put("STOP")  

    def media(self):
        with open(self.datapath, 'rb') as fin:
            for offset in iter(self.inq.get , "STOP"):
                try:
                    fin.seek(offset)
                    tweet = fin.readline()
                    tweet_obj = json.loads(tweet)
                    if self.noretweet and is_retweet(tweet_obj, include_old=(not self.no_oldr)):
                        continue
                    if self.english_only and tweet_obj.get("lang", None) != 'en':
                        continue
                    # geofilter
                    geotag = None
                    if self.geofilter:
                        geotag = geofilter(tweet_obj, self.shape)
                        if geotag == None:
                            continue
                    text = preprocess(tweet_obj, self.stopword)
                    kws = set(has_kw(text, self.track))
                    media = has_media(tweet_obj, self.instagram, self.photo_only) 
                    self.outq.put((tweet_obj['id_str'], kws, geotag, media))
    
                except ValueError as e:
                    continue
                except Exception as e:
                    print traceback.format_exc()
        self.outq.put("STOP")
        
    def sentiment(self):
        import string
        with open(self.datapath, 'rb') as fin:
            for offset in iter(self.inq.get , "STOP"):
                try:
                    fin.seek(offset)
                    tweet = fin.readline()
                    tweet_obj = json.loads(tweet)
                    if self.noretweet and is_retweet(tweet_obj, include_old=(not self.no_oldr)):
                        continue
                    if self.english_only and tweet_obj.get("lang", None) != 'en':
                        continue
                    # geofilter
                    geotag = None
                    if self.geofilter:
                        geotag = geofilter(tweet_obj, self.shape)
                        if geotag == None:
                            continue
                    text = preprocess(tweet_obj, self.stopword)
                    kws = set(has_kw(text, self.track))
                    text = ' '.join(map(lambda x: x.lstrip(string.punctuation), text.split()))
                    self.outq.put((tweet_obj['id_str'], kws, geotag, text))
                except ValueError as e:
                    continue
                except Exception as e:
                    print traceback.format_exc()
        self.outq.put("STOP")

    def count_retweet(self):
        with open(self.datapath, 'rb') as fin:
            for offset in iter(self.inq.get , "STOP"):
                try:
                    fin.seek(offset)
                    tweet = fin.readline()
                    tweet_obj = json.loads(tweet)
                    if self.noretweet and is_retweet(tweet_obj, include_old=(not self.no_oldr)):
                        continue
                    if self.english_only and tweet_obj.get("lang", None) != 'en':
                        continue
                    # geofilter
                    geotag = None
                    if self.geofilter:
                        geotag = geofilter(tweet_obj, self.shape)
                        if geotag == None:
                            continue
                    count = 0
                    # how many times the original tweet has been retweeted
                    if is_retweet(tweet_obj, include_old=(not self.no_oldr)):
                        count = tweet_obj.get('retweeted_status', None)
                        if count == None:
                            # if old style retweet
                            if not self.no_oldr:
                                count = tweet_obj.get('retweet_count', 0)
                            else:
                                count = 0
                        else:
                            count = count.get('retweet_count', 0)
                    else:
                        count = tweet_obj.get('retweet_count', 0)

                    text = preprocess(tweet_obj, self.stopword)
                    kws = set(has_kw(text, self.track))
                    self.outq.put((tweet_obj['id_str'], kws, geotag, count))

                except ValueError as e:
                    continue
                except Exception as e:
                    print traceback.format_exc()
        self.outq.put("STOP")

    def df(self):
        """
        total_tf estimated here has no importance: do not use it
        """
        with open(self.datapath, 'rb') as fin:
            for offset in iter(self.inq.get , "STOP"):
                try:
                    fin.seek(offset)
                    tweet = fin.readline()
                    tweet_obj = json.loads(tweet)
                    if self.noretweet and is_retweet(tweet_obj, include_old=(not self.no_oldr)):
                        continue
                    if self.english_only and tweet_obj.get("lang", None) != 'en':
                        continue
                    # geofilter
                    geotag = None
                    if self.geofilter:
                        geotag = geofilter(tweet_obj, self.shape)
                        if geotag == None:
                            continue
                    text = preprocess(tweet_obj, self.stopword)
                    #total_tf = len(text.split())
                    kws = set(has_kw(text, self.track))
                    total_tf = len(kws) + (len(text.split()) - sum(map(lambda x: len(x.split()), kws)))
                    #kws = set(filter(lambda x: x in self.track, text.split()))
                    self.outq.put((tweet_obj['id_str'], kws, total_tf, geotag))
                except ValueError as e:
                    continue
                except Exception as e:
                    print traceback.format_exc()
        self.outq.put("STOP")
    def profiler(self):
        """total tweets, retweet, english, geotagged tweets"""
        with open(self.datapath, 'rb') as fin:
            for offset in iter(self.inq.get , "STOP"):
                try:
                    profile={"total":0, "retweets-total": 0, "retweets-english":0, "retweets-geotagged":0, "non-retweets-total":0, "non-retweets-english":0, "non-retweets-geotagged":0}
                    fin.seek(offset)
                    tweet = fin.readline()
                    tweet_obj = json.loads(tweet)
                    profile['total'] += 1
                    geotag = None
                    if self.geofilter:
                        geotag = geofilter(tweet_obj, self.shape)
                        if geotag == None:
                            continue
                    state = "non-retweets"
                    if is_retweet(tweet_obj, include_old=(not self.no_oldr)):
                        state = "retweets"
                    profile[state + '-total'] += 1
                    profile[state +'-english'] += int(tweet_obj.get("lang", None) == 'en')
                    profile[state + '-geotagged'] += int(geofilter(tweet_obj, None) != None)
                    cprofile = Counter({k:Counter(v) if isinstance(v, dict) else v for k, v in profile.items()})
                    self.outq.put((tweet_obj['id_str'], cprofile))
                except ValueError as e:
                    continue
                except Exception as e:
                    print traceback.format_exc()
        self.outq.put("STOP")  
    def term_correlation(self):
        """What are the terms that co-occur with given term: DF like"""
        with open(self.datapath, 'rb') as fin:
            for offset in iter(self.inq.get , "STOP"):
                try:
                    fin.seek(offset)
                    tweet = fin.readline()
                    tweet_obj = json.loads(tweet)
                    if self.noretweet and is_retweet(tweet_obj, include_old=(not self.no_oldr)):
                        continue
                    if self.english_only and tweet_obj.get("lang", None) != 'en':
                        continue
                    # geofilter
                    geotag = None
                    if self.geofilter:
                        geotag = geofilter(tweet_obj, self.shape)
                        if geotag == None:
                            continue
                    text = preprocess(tweet_obj, self.stopword)
                    kws = set(has_kw(text, self.track))
                    ngrams = set(find_ngrams(text, self.ngram)) 
                    self.outq.put((tweet_obj['id_str'], kws, ngrams, geotag))

                except ValueError as e:
                    continue
                except Exception as e:
                    print traceback.format_exc()

        self.outq.put("STOP")       
    def most_common(self):
        """What are the terms that co-occur with given term: DF like"""
        with open(self.datapath, 'rb') as fin:
            for offset in iter(self.inq.get , "STOP"):
                try:
                    fin.seek(offset)
                    tweet = fin.readline()
                    tweet_obj = json.loads(tweet)
                    if self.noretweet and is_retweet(tweet_obj, include_old=(not self.no_oldr)):
                        continue
                    if self.english_only and tweet_obj.get("lang", None) != 'en':
                        continue
                    # geofilter
                    geotag = None
                    if self.geofilter:
                        geotag = geofilter(tweet_obj, self.shape)
                        if geotag == None:
                            continue
                    text = preprocess(tweet_obj, self.stopword)
                    ngrams = find_ngrams(text, self.ngram)
                    self.outq.put((tweet_obj['id_str'], ngrams, geotag))

                except ValueError as e:
                    continue
                except Exception as e:
                    print traceback.format_exc()

        self.outq.put("STOP") 
if __name__ == "__main__":
    ### Command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s' , '--subroutine',
             help="Name of the process to be parallelized \
                        values = ['tf', 'df', 'm(count media)', 'cr(count retweeted)', 's(sentiment)', 'tc(term correlation)', 'mc(most common)', 'gp(global profiler)'], default=tf", 
                        default="tf")
    parser.add_argument('-d' , '--datapath', help="Data file or directory")
    parser.add_argument('-o' , '--output', 
            help="output file path")
    parser.add_argument('-kw' , '--kw', help="a list of keywords", 
                        default="keyword_terms.txt")
    parser.add_argument('-lo', '--line_offsets', help="Path to pickled line offsets. default:None",
                        default=None)
    parser.add_argument('-xr', '--noretweet', help="do not include retweets", 
                        action='store_true')
    parser.add_argument('-xoldr', '--no_oldr', help="do not include old style retweets", 
                        action='store_true')
    parser.add_argument('-en', '--english_only', help="analyze only english tweets", 
                       action="store_true")
    parser.add_argument('-gf', '--geofilter' , help='geo resolutions to filter \
                            valid values : [us, states, geotagged, custom] if custom, \
                            shapefile has to be specified default=None', 
                            default=None)
    parser.add_argument('-sp', '--shapefile', help="used only when  \
                            --geofilter is custom. default=None", default=None)
    parser.add_argument('-p', '--photo_only', help="report only photo \
                        entity fraction, default=False", action="store_true")
    parser.add_argument('-i', '--instagram', help="include instagram \
                            link in media count, default=False", action='store_true')
    parser.add_argument('-tk', '--topk', help="Topk terms to report correlation default=10",
                               default=10)
    parser.add_argument('-n', '--ngram', help="Ngram length; subroutine must be \
                                tc-term correlation, default=1", default=1)
    args = parser.parse_args()

    _start = time.time()
        
    #if None in [args.subroutine, args.datapath, args.kw]:
    if not all(map(lambda x: x != None, [args.subroutine, args.datapath, args.kw])):
        print "Some required parameters are missing"
        sys.exit()
    if args.geofilter and args.geofilter.strip().lower() != 'custom' and args.shapefile:
        print "WARNING: shapefile will not be used. specifiy custom in geofilter"
        sys.exit()
    if str(args.geofilter).strip().lower() not in ['us', 'states', 'geotagged', 'custom', 'none']:
        print "Invalid value in geofilter, Quitting!"
        sys.exit()
    workers = Workers(args.subroutine, args.datapath, args.output, args.kw, args.line_offsets, 
                    args.noretweet, args.no_oldr, args.english_only, 
                    args.geofilter, args.shapefile,
                    args.photo_only, args.instagram,
                    args.topk, args.ngram) 

    workers.start()
    print time.time() - _start 
