# Sentiment Analysis Engine for Tweet sentiment classification with Naive Bayes Classifier
Creator: Bahram Vazirnezhad
Date: 8Jul2017

## First Version Specifications
Designed/Developed in Python
Learns from samples of +ve and -ve tweets
Learns lexicon of valid words automatically from the most popular words
Uses bag of words: counts but order and syntax of tweets are not considered
Machine Learning: NB classifier- with smoothing for unseen data
Preprocessing of tweets for Lower casing, Stemming, Stop words removal, Negation detection, Analysis of contracted forms of verb negation

## Baseline Algorithm 
Tokenization
Feature Extraction: Bag of words after pre processing
Stop word removal
Negation: Add (Not_) to every word between negation and following punctuation:
Stemming with Snowball stemmer
Automatic lexicon building by sorting tokens based on their frequency
Unknow words are identified as Out of Vocabulary words and are replaced by UNK sign in train and use
Naïve Bayes Classifier 

## Usage
For executing the codes you need to store tweets in the format of example sample files in csv format and
run training.py. All preprocessing functions are in processing folder within class processing.

##Evaluation
Training was done over 1 million tweets chosen randomly from the provided file 
Test was done over 10010 tweets from held out set which was chosen randomly from the provided file. 
None of the tweets from the test set have not been used for training
Accuracy: 76% (number of correctly identified sentiments devided by total number of test samples)

## feedbacks and questions
Please send your feedbacks and questions to creator email: bahram@live.com.au
 
 
  
 
 
  
 





