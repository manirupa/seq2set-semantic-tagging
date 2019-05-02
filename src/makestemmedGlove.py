"""
Script to create a stemmed Glove word embedding file using the Porter Stemmer
"""
from nltk.stem import *
import os, sys, re

ps = PorterStemmer()
eg = ps.stem('jumping')
#eg
#u'jump'
print(str(eg))
#'jump'

def write_to_file(str_to_print,filename):
    f = open(filename, 'a')
    f.write(str_to_print)
    f.close()
    
filepath = 'data/glove.6B.50d.txt'
stemmed_filepath = 'data/stemmed.glove.6B.50d.txt'
stemmed_dict = {}
count = 1
with open(filepath) as fp:  
   for line in fp:
       tokens = line.split()
       tokens[0] = str(ps.stem(tokens[0]))
       stemmed_word = tokens[0]
       stemmed_dict[stemmed_word] = stemmed_dict.get(stemmed_word, 0) + 1
       stemmed_line = ' '.join(tokens)
       print(count, tokens[0], tokens[1], tokens[2])
       if(stemmed_dict[stemmed_word] == 1): #write only once per key
           write_to_file('%s\n' % stemmed_line, stemmed_filepath)
       count += 1
       if (count % 100 == 0):
           print(count, line)

print(len(stemmed_dict), stemmed_dict)