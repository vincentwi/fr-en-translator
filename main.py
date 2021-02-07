import numpy as np 
import pandas as pd 
import math, copy, time
from fastai.text.all import *
from torchtext import data, datasets
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline


path = '../input/language-translation-englishfrench/eng_-french.csv'

df = pd.read_csv(path)


#fastai not working so we r gonna use the actual library
import spacy
spacy_en = spacy.load('en')
spacy_fr = spacy.load('fr')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]




english_words_counter = collections.Counter([word for sentence in df['English words/sentences'] for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in df['French words/sentences'] for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in df['English words/sentences'] for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in df['French words/sentences'] for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')






