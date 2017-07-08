# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:24:07 2017
Version 0.1 Finalized
@author: Bahram
"""
# this code gets CSV file and gets tweets and labels to feed a NB classifier for computing parameters if classifier
import pickle
import csv
import tokenizer
from trainer import Trainer
from operator import itemgetter
from classifier import Classifier
from preprocess.preprocess import preprocess # to get tokens, stem and remove stop words and negation detection.
tweetTrainer = Trainer(tokenizer.Tokenizer(stop_words = [], signs_to_remove = ['?!#%&']))#
process = preprocess()

def processing (training):
    # preprocess sentences for stemming, removing stop words and negation detection and apply
    trainingProcessed = list(map(process.gettokens, training))
    trainingProcessed = list(map(process.stemtokens, trainingProcessed))
    trainingProcessed = list(map(process.removestopwords, trainingProcessed))
    #print (trainingProcessed)
    trainingProcessed = list(map(process.negatesequence, trainingProcessed))
    trainingProcessedWords = []
    for sentence in trainingProcessed:
        sentence = ' '.join(sentence)
        wordsProcessed = sentence.split()
        for wordProcessed in wordsProcessed:
            trainingProcessedWords.append(wordProcessed)
    return trainingProcessedWords

def unkWord (train, lexicon):
    # this function replace unk-known tokens with UNK in training and use
    i = 0
    for token in train:
        if token not in lexicon:
            train [i] = 'UNK'
        i = i + 1
    return train
# here the negative and positive tweets are being read and stored in lists   
trainingSet0 = []
trainingSet1 = []    
num = 0
file=open('Sentiment_Analysis_Dataset_Social_Shuffled.csv', 'r', encoding="Latin-1")#.csv
reader = csv.reader(file)
for line in reader:
    if line [1] == '0':
        trainingSet0.append(line[3])
        num = num + 1
    elif line [1] == '1':
        trainingSet1.append(line[3])
        num = num + 1
    if num > 1000000:
        break
# tweets are processing
trainingSet0 = processing(trainingSet0)
trainingSet1 = processing(trainingSet1)
# here a lexicon of first 3000 more common tokens are stored in lexicon,
# all tokens out of this list are considered as UNK token
allWords = trainingSet0 + trainingSet1
wordCounter = {}
for word in allWords:
    if word in wordCounter:
        wordCounter[word] += 1
    else:
        wordCounter[word] = 1
popularWords = sorted(wordCounter, key = wordCounter.get, reverse = True)
lexicon = popularWords[:4000]
 # After learning lexicon here OOV words are replaced by UNK sign
trainingSet0 = unkWord(trainingSet0, lexicon)
trainingSet1 = unkWord(trainingSet1, lexicon)
# positive and negative tweets are passed to training onject
for word in trainingSet0:
    tweetTrainer.train(word, '0')
for word in trainingSet1:
    tweetTrainer.train(word, '1')
    
# a classifier instance
sentimentClassifier = Classifier(tweetTrainer.data, tokenizer.Tokenizer(stop_words = [], signs_to_remove = []))
# storage of lexicon and sentiment classifier on disk
c = open('sentimentClassifier.pickle', 'wb')
l = open('lexicon.pickle', 'wb')
pickle.dump(sentimentClassifier, c)
pickle.dump(lexicon, l)
c.close()
l.close()
file.close()

# test section
# loading lexicon and sentimentClassifier for evaluation over random samples
# samples are chosen randomly and not within training samples
c = open('sentimentClassifier.pickle', 'rb')
l = open('lexicon.pickle', 'rb')
fileEval=open('test2.csv', 'r', encoding="Latin-1")#.csv
sentimentClassifier = pickle.load(c)
lexicon = pickle.load(l)
reader = csv.reader(fileEval)
realClass = []
recClass = []
for line in reader:
    realClass.append (line [1])
    instance = [line [3]]
    instance = processing(instance)
    instance = unkWord(instance, lexicon)
    instance = ' '.join(instance)
    classification = sentimentClassifier.classify (instance)
    recClass.append(max(classification,key=itemgetter(1))[0])   
# computing accuracy by comparing real classes with recognized classes
cnt = 0
i = 0
for cls in recClass:
    if cls == realClass[i]:
        cnt = cnt + 1
    i = i + 1
accuracy = cnt/(len(realClass))
print (accuracy)
fileEval.close()