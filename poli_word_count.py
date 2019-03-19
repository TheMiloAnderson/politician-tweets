import pandas as pd
# import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from token_maker import get_string
from utils import parse_sentence
from utils import classify_model
from utils import classify_sentence
from utils import setup_database
from utils import add_to_database

nltk.downloader.download('vader_lexicon')

setup_database()

clf = classify_model()
learn_response = 0


def class_maker(H):
    subj = set()
    obj = set()
    verb = set()
    triples, root = parse_sentence(H)
    triples = list(triples)

    for t in triples:
        if t[0][1][:2] == 'VB':
            verb.add(t[0][0])
        relation = t[1]
        if relation[-4:] == 'subj':
            subj.add(t[2][0])
        if relation[-3:] == 'obj':
            obj.add(t[2][0])

    print("\t" + "Subject: " + str(subj) + "\n" + "\t" + "Object: " + str(obj) + "\n" + "\t" + "Topic: " + str(
        root) + "\n" + "\t" + "Verb: " + str(verb))
    subj = list(subj)
    obj = list(obj)
    verb = list(verb)
    proper_nouns = set()

    for t in triples:
        if t[0][1] == 'NNP':
            proper_nouns.add(t[0][0])
        if t[2][1] == 'NNP':
            proper_nouns.add(t[2][0])
    proper_nouns == list(proper_nouns)
    print("\t" + "Proper Nouns: " + str(proper_nouns))

    # classification
    classification = classify_sentence(clf, H)
    print(classification)
    import pdb; pdb.set_trace()
    add_to_database(classification, subj, root, verb, H)


class WordCount():
    def __init__(self):
        self.df = pd.read_csv('words/ExtractedTweets.csv')
        self.word_dict = {}

        self.word_dict_build()

    def word_score(self, list_thing):
        self.pos_word_dict = {}
        self.neu_word_dict = {}
        self.neg_word_dict = {}
        self.sid = SentimentIntensityAnalyzer()
        self.word_score_dict = {'pos': [], 'neu': [], 'neg': []}

        for word in list_thing:
            if (self.sid.polarity_scores(word)['compound']) >= 0.4:
                if word not in self.pos_word_dict.keys():
                    self.pos_word_dict[word] = 1
                else:
                    self.pos_word_dict[word] += 1
            elif (self.sid.polarity_scores(word)['compound']) <= -0.4:
                if word not in self.neg_word_dict.keys():
                    self.neg_word_dict[word] = 1
                else:
                    self.neg_word_dict[word] += 1
            elif word not in self.neu_word_dict.keys() and word not in self.neg_word_dict.keys() and word not in self.pos_word_dict.keys():
                self.neu_word_dict[word] = 1
            else:
                self.neu_word_dict[word] += 1
        self.word_score_dict['pos'] = self.pos_word_dict
        self.word_score_dict['neu'] = self.neu_word_dict 
        self.word_score_dict['neg'] = self.neg_word_dict
        print(self.word_score_dict)

        return self.word_score_dict

    def word_dict_build(self):
        for tweet_index in range(len(self.df.Tweet)):
            self.word_lst = [i.lower() for i in self.df.Tweet[tweet_index].split(' ')]
            self.thing = get_string(tweet_index, self.df.Tweet[tweet_index])
            self.classy = class_maker(self.df.Tweet[tweet_index])
            self.word_dict['Tweet #' + str(tweet_index)] = [self.thing, self.word_score(self.word_lst)]
            for i in range(len(self.word_lst)):
                try:
                    if self.word_lst[int(i)] == '#':
                        self.word_dict['#'] += 1
                except:
                    pass
                    # import pdb; pdb.set_trace()
                else:
                    if self.word_lst[i] in self.word_dict.keys():
                        self.word_dict[self.word_lst[int(i)]] += 1
                    else:
                        self.word_dict[self.word_lst[int(i)]] = 1

        # print(self.word_dict)
        return self.word_dict


dict_thing = WordCount()


# from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
# print(word_lst)
# hash_vec = HashingVectorizer(n_features=2**4)
# tfd_vec = TfidfVectorizer()
# tf = tfd_vec.fit(word_lst)
# vector = tfd_vec.transform([word_lst[0]])
# print(tf)
# print(vector.shape)
# X = tfd_vec.fit_transform(word_lst)
# print(X.shape)
