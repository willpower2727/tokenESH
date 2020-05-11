# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 07:09:59 2020

Test script of pulling in requirements and comparing user input to requirements using 
TFIDF and Cosine Similarity

@author: Will
"""

import nltk
#import numpy as np
import random
import string
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

########Get requirements into dictionary
raw = open('Requirements.txt','r')
reqs = csv.reader(raw,delimiter='\r')
d = list(reqs)
reqdict = {}
for i in range(0,len(d)):
    reqdict[str(d[i]).split(':')[0]] = str(d[i]).split(':')[1]
    #print(d[i])
#print(reqdict)

senttokens = {}
for key in reqdict:
    senttokens[key] = nltk.sent_tokenize(reqdict[key])



####setup some tools for chatbot
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ["hello", "hi", "greetings", "sup","what's up", "hey"]

GREETING_RESPONSES = ["hello"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
def response(user_response):
    #robo_response = ''
    senttokens['user_response'] = [user_response]
    print(senttokens['user_response'])
    tfidf = {}
    vals = {}
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    # for key in senttokens:
    #     tfidf[key] = TfidfVec.fit_transform(senttokens[key])
    #tfidf = TfidfVec.fit_transform(senttokens)
    #print(tfidf)
    for key in tfidf:
        #vals[key] = cosine_similarity(tfidf[key], tfidf['user_response'])
        tfidf = TfidfVec.fit_transform(senttokens[key])
        print('fitting key: ', key)
        print(tfidf)
        vals[key] = cosine_similarity(TfidfVec.fit_transform(senttokens['user_response']), tfidf)
        print(vals)
    #print(vals)
    #idx = vals.argsort()[0][-2]
    #flat = vals.flatten()    
    #flat.sort()
    #req_tfidf = flat[-2]
    
    #if (req_tfidf==0):
    #    robo_response = robo_response+"I don't know"
    #    return robo_response
    #else:
    #    robo_response = robo_response+sent_tokens[idx]
    return vals

flag = True
print("ROBO: My name is Robo. If you want to exit, type bye")

while (flag==True):
    user_response = input()
    user_response = user_response.lower()
    
    if (user_response != 'bye'):
        if (user_response== 'thanks' or user_response== 'thank you'):
            flag = False
            print("ROBO You are welcome..")
        else:
            if (greeting(user_response) != None):
                print("ROBO: " +greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                #senttokens.remove('user_response')
    else:
        flag = False
        print("ROBO: Bye! take care..")
        
        
        
        
        
        
        
        
        
        
        
















