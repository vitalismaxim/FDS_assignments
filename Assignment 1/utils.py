import re
from datetime import date
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import nltk
import pandas as pd
from nltk.probability import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from utils import * 


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

import config as cf

def preprocess(col, steps = ['prep_lower', 
                             'prep_tokenizewords', 
                             'prep_drop_sentenceindicators', 
                             'prep_drop_numbers', 
                             'prep_drop_stopwords',
                      'prep_listtosentence']):
    if 'prep_lower' in steps:
        col = col.str.lower()
    if 'prep_tokenizewords' in steps:
        col = col.apply(word_tokenize)
    if 'prep_drop_sentenceindicators' in steps:
        col = col.apply(prep_drop_sentenceindicators)
    if 'prep_drop_numbers' in steps:
        col = col.apply(prep_drop_numbers)
    if 'prep_drop_stopwords' in steps:
        col = col.apply(prep_drop_stopwords)
    if 'shorten_texts' in steps:
        print('not implemented yet')
    if 'prep_listtosentence' in steps:
        col = col.apply(lambda x: ' '.join(x))
        
    return col
    


def prep_drop_sentenceindicators(col):
    sentenceindicators = [',', '.', '?', '\t', ':', ';', "''", "'", '"',]
    col = [word for ll in col for word in ll if word not in sentenceindicators]

    return col

def prep_drop_numbers(s):
    s = [i for i in s if not i.replace('.', '').isdigit()]

    return s

def prep_drop_stopwords(words):
    sw = stopwords.words("english")
    words = [word for word in words if word not in sw]
    return words


def create_wordcloud():
    def cloud(text, filename):
        # Create stopword list:
        stopwords = set(STOPWORDS)

        # Add some words to the stop word list, this can be adjusted
        stopwords.update(["â", "see", "going", "u", "thank", ',',
                          "you", "itâ", "s", "well", "us", "weâ",
                          "will", "continue", 'hello', 'Hello', 'Good afternoon', 'afternoon',
                          "now", "re", 'thank you', 'thanks', 'Thank', 'Thanks', 'good morning', 'morning'])

        # Generate a word cloud image
        wordcloud = WordCloud(width=600, height=400, max_font_size=90,
                              max_words=50, stopwords=stopwords,
                              background_color="white").generate(text)

        # Display the generated image
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

        # Save the wordcloud
        wordcloud.to_file(join(cf.path_images, filename))

        return None

    print('Generating wordclouds for presentations and Q&A sessions')

    # Load data
    texts = pd.read_pickle(cf.A_B_texts_and_prices_file)

    # Create word cloud for presentations
    text = ""
    for t in texts.presentation:
        if str(t) != 'nan':
            text += ' ' + t
    text = re.sub('\n', ' ', text)
    cloud(text, "wordcloud_presentations.png")


    print('Finished generating wordclouds for presentations and Q&A sessions')

    return None