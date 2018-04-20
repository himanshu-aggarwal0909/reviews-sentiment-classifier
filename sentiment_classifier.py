################################################## 
from sklearn import *
import nltk
from nltk import *
import numpy as np
import random
import itertools
import statistics
import pandas as pd
import string
from nltk.stem import PorterStemmer  #Stemmer used to get the stem/root ofr the word
from nltk.classify.scikitlearn import SklearnClassifier # A wrapper to include the scikit learn algorithms within the nltk classifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from sklearn.svm import SVC, LinearSVC,  NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle , csv , random
from sklearn.metrics import confusion_matrix 

stop_words = stopwords.words('english')
ps = PorterStemmer()




def remove_puncutation(word):
    no_punct = ""
    for char in word:
       if char not in string.punctuation:
        no_punct = no_punct + char
    word = no_punct
    return word

# Input : take the word
# Output : returns the stem of the word 
def stem_the_word(word):
	word.encode('utf-8')
	return ps.stem(word)

# Purpose : To get the word count (Frequency Distribution) for top 1000 most common words 
def find_word_count(words):
    words = FreqDist(words)
    word_features = [w for (w, c) in words.most_common(1000)]
    return word_features

#Input : list of tuples of 2 elements having filtered_tokenize_words and sentiment [(filtered_tokenize_words),(sentiment)]
#Output : list of all words in the data
#Purpose : To get the list of all words in each review 
def find_words_in_data(filtered_training_data):
    all_words = []
    for (words, sentiment) in filtered_training_data:
        if words != "":
          all_words += words
    return all_words

#Input: list of lines of result.txt
#Output : list of tuples of 2 elements having tokenize_words and sentiment [(tokenize_words),(sentiment)]
#Purpose : This Function Divides the Training Data  in to a list of tuples of 2 elements having tokenize_words and sentiment [(tokenize_words),(sentiment)]	
def tokenize_the_review_training_data(review_data):
	tokenize_training_data = []
	for each_review in review_data:
		sentiment = each_review.split(',')[0]
		review_text = str(each_review.split(',')[1:])
		tokenize_training_data.append((word_tokenize(review_text), sentiment))
	return tokenize_training_data


#Input :  list of tuples of tokenize_words and sentiment
#Output : list of tuples of filtered_words and sentiment
#Purpose : removes the words which are in stopwords and also do the stemming of the words and also removes the punctuation
def filter_words(tokenize_data):
	filtered_word_data = []
	for (tokenize_words, sentiment) in tokenize_data:
		#filtered_word_data.append(([stem_the_word(remove_puncutation(word.lower()).encode('utf-8')) for word in tokenize_words if remove_puncutation(word.lower()).decode('utf-8','ignore') not in stop_words and remove_puncutation(word.lower()).decode('utf-8','ignore') != ''], sentiment))
		filtered_word_data.append(([stem_the_word(remove_puncutation(word.lower())) for word in tokenize_words if remove_puncutation(word.lower()) not in stop_words and remove_puncutation(word.lower()) != ''], sentiment))
		'''
		words_filtered = []
		for word in tokenize_words:
			if remove_puncutation(word.lower()).decode('utf-8','ignore') not in stop_words:
				words_filtered.append(stem_the_word(remove_puncutation(word.lower()).encode('utf-8')))
		while ' ' in words_filtered:
			words_filtered.remove(' ')
		while '' in words_filtered:
			words_filtered.remove('')
     	filtered_word_data.append((words_filtered, sentiment))
     	for word in words_filtered:
        correct_words.append(stem_words(correct(word)))
    	review_training_set.append((correct_words, sentiment))
'''
	return filtered_word_data

def freq_count(tokens):
	return nltk.FreqDist(tokens).most_common(50)

# It extracts the features take the all the words present in the entire data and add a TRUE if it is present in that particular or otherwise false
def find_features(each_review_words):
	global all_words_in_data
	features = {}
	words = set(each_review_words)
	for each_word in word_features_set:
		features[each_word] = (each_word in words)
		'''if each_word in words:
			features[each_word] = 1
		else:
			features[each_word] = 0
			'''
	return features

def storing_dataset(training_set):
    writer=csv.writer(open("dataset.csv",'w'))
    header = word_features
    header.append("CLASS_SENTIMENT")   #  1 - for positive   and  2 - for negative
    writer.writerow(header)
    for each_row in training_set:
        list = []
        dict = each_row[0]
        lable = each_row[1]
        for key in dict:
          list.append(dict[key])
        list.append(lable)
        writer.writerow(list)



training_file = open('Reviews.txt')
review_lines = training_file.readlines()
tokenize_training_data = tokenize_the_review_training_data(review_lines)
filtered_training_data = filter_words(tokenize_training_data)

word_features =find_word_count(find_words_in_data(filtered_training_data))
word_features_set = set(word_features)

training_set = [(find_features(d), c) for (d,c) in filtered_training_data]

storing_dataset(training_set)



#print(training_set)
#random.shuffle(training_set)
train_set = training_set[:3500]
test_set = training_set[3500:]
random.shuffle(train_set)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("LinearSVC_Classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)

Naive_classifier = NaiveBayesClassifier.train(train_set)
print("Naive_Classifier accuracy percent: ", (nltk.classify.accuracy(Naive_classifier, test_set))*100)

nn_classifier = SklearnClassifier(KNeighborsClassifier())
nn_classifier.train(train_set)
print("nn_Classifier accuracy percent: ", (nltk.classify.accuracy(nn_classifier, test_set))*100) 
print(30*'$')


class majority_classifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		results = []
		for c in self._classifiers:
			res = c.classify(features)
			results.append(res)
		return statistics.mode(results)

	def confidence(self, ):
		results = []
		for c in self._classifiers:
			res = c.classify(features)
			results.append(res)
		favour_results = results.count(statistics.mode(results))
		conf_idence = favour_results / len(results)
		return conf_idence

compund_classifier = majority_classifier(LinearSVC_classifier, Naive_classifier, nn_classifier)

y_pred = []
y_actual = []
for each_set in test_set:
	y_actual.append(each_set[1])
	y_pred.append(compund_classifier.classify(each_set[0]))
print(y_pred)
print(y_actual)
#confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_actual, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1],
                      title='Confusion matrix, without normalization')
plt.show()