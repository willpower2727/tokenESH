# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:28:36 2020

@author: Will
"""

import urllib.request  
import re
#import random
import string 
import bs4 as bs
import nltk
import sklearn

#raw_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming')  
raw_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Miscegenation')
raw_data = raw_data.read()
html_data = bs.BeautifulSoup(raw_data,'lxml')
all_paragraphs =html_data.find_all('p')
article_content = ""
for p in all_paragraphs:  
    article_content += p.text

article_content =  article_content.lower()# converts to lowercase
#print(article_content)
article_content  = re.sub(r'\[[0-9]*\]', ' ', article_content )  
article_content = re.sub(r'\s+', ' ', article_content )  
sentence_list = nltk.sent_tokenize(article_content)  
article_words= nltk.word_tokenize(article_content )

nltk.download('punkt') 
nltk.download('wordnet') 
 
lemmatizer = nltk.stem.WordNetLemmatizer()
def LemmatizeWords(words):
    return [lemmatizer.lemmatize(word) for word in words]
remove_punctuation= dict((ord(punctuation), None) for punctuation in string.punctuation)
 
def RemovePunctuations(text):
    return LemmatizeWords(nltk.word_tokenize(text.lower().translate(remove_punctuation)))
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dummy_user_input = 'global warming is awesome'

def give_reply(user_input):
    #chatbot_response=''
    sentence_list.append(user_input)
    print(sentence_list)
    word_vectors = TfidfVectorizer(tokenizer=RemovePunctuations, stop_words='english')
    vectorized_words = word_vectors.fit_transform(sentence_list)
    #print(vectorized_words)
    similarity_values = cosine_similarity(vectorized_words[-1], vectorized_words)
    #print(similarity_values)
    similar_sentence_number =similarity_values.argsort()[0][-2]
    print("sorted similarity values: ", similarity_values.argsort())
    print("similar sentence number ", similar_sentence_number)
    similar_vectors = similarity_values.flatten()
    similar_vectors.sort()
    matched_vector = similar_vectors[-2]
    return sentence_list[similar_sentence_number]

best_match = give_reply(dummy_user_input)
#print(remove_punctuation)
print(best_match)









