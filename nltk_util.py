import nltk 
import numpy as np

#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemer = PorterStemmer()

def tokenization(sentence):
    return nltk.word_tokenize(sentence)

def stemm(word):
    return stemer.stem(word.lower())


def bag_of_word(tokenized_sentence, all_words):
      
    tokenized_sentence = [stemm(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype= np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag 

