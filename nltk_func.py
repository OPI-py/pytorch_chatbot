import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    '''Splitting a string into meaningful units'''
    return nltk.word_tokenize(sentence)
    
def stem(word):
    '''Remove morphological affixes from words, leaving only the word stem'''
    return stemmer.stem(word.lower())
    
def bag_of_words(tokenized_sentence, all_words):
    '''Convert the pattern strings to numbers
    that the network can understand'''
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    # make array with zeros for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    # replace 0 with 1 if word present in sentence
    for  index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1
    return bag
