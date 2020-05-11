# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a test chatbot in python
"""

import nltk
import numpy as np
import random
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f=open('chatbot.txt','r',errors='ignore')

raw = f.read()

#print(raw)

raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
#print(word_tokens)

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
    robo_response = ''
    sent_tokens.append(user_response)
    
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    print(tfidf)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()    
    flat.sort()
    req_tfidf = flat[-2]
    
    if (req_tfidf==0):
        robo_response = robo_response+"I don't know"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

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
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! take care..")




























