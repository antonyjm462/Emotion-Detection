# imports
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
import unicodedata
import pandas as pd 
from pandas import DataFrame, read_csv
import os
import csv 
import numpy as np 


train_path = "aclImdb/train/" # source data
test_path = "aclImdb/test/" # test data
outpath ="./normailized_data/"



nlp = spacy.load('en_core_web_sm', parse = False, tag=False, entity=False)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
special_char_pattern = re.compile(r'([{.(-)!}])')


# # Cleaning Text - strip HTML
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


# # Removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# # Expanding Contractions
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
                                   if contraction_mapping.get(match) \
                                    else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# # Removing Special Characters
def remove_special_characters(text):
	text = re.sub('[^a-zA-Z0-9\s]', '', text)
	text = re.sub(' +', ' ', text)
	return text


# # Lemmatizing text
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# # Removing Stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# remove extra newlines
# insert spaces between special characters to isolate them    
def clean(text):
	text = re.sub(r'[\r|\n|\r\n]+', ' ',text) 
	text = special_char_pattern.sub(" \\1 ", text)
	return text

def normalize(filename,data):
	pol = filename.split("_")
	pol = pol[1].replace(".txt","")
	data = strip_html_tags(data)
	data = remove_accented_chars(data)
	data = expand_contractions(data)
	data = data.lower()
	data = clean(data)
	data = lemmatize_text(data)
	data = remove_special_characters(data) 
	data = remove_stopwords(data, is_lower_case=True)
	print(f"data: {data} pol:{pol}")
	return data,pol

	

def imdb_data_preprocess(inpath, outpath, name, mix):
	stopwords = open("stopwords.en.txt", 'r' , encoding="ISO-8859-1").read()
	stopwords = stopwords.split("\n")

	indices = []
	text = []
	rating = []

	i =  0 

	for filename in os.listdir(inpath+"pos"):	
		data = open(inpath+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
		data,pol = normalize(filename,data)
		indices.append(i)
		text.append(data)
		rating.append(pol)
		i = i + 1

	for filename in os.listdir(inpath+"neg"):
		pol = filename.split("_")
		pol = pol[1].replace(".txt","")
		data = open(inpath+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
		data,pol = normalize(filename,data)
		indices.append(i)
		text.append(data)
		rating.append(pol)
		i = i + 1

	Dataset = list(zip(indices,text,rating))
	
	if mix:
		np.random.shuffle(Dataset)

	df = pd.DataFrame(data = Dataset, columns=['row_number', 'text', 'polarity'])
	df.to_csv(outpath+name, index=False, header=True)

	pass


# '''
# REMOVE_STOPWORDS takes a sentence and the stopwords as inputs and returns the sentence without any stopwords 
# Sentence - The input from which the stopwords have to be removed
# Stopwords - A list of stopwords  
# '''
# def remove_stopwords(sentence, stopwords):
# 	sentencewords = sentence.split()
# 	resultwords  = [word for word in sentencewords if word not in stopwords]
# 	result = ' '.join(resultwords)
# 	return result


'''
UNIGRAM_PROCESS takes the data to be fit as the input and returns a vectorizer of the unigram as output 
Data - The data for which the unigram model has to be fit 
'''
def unigram_process(data):
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer()
	vectorizer = vectorizer.fit(data)
	return vectorizer	


'''
BIGRAM_PROCESS takes the data to be fit as the input and returns a vectorizer of the bigram as output 
Data - The data for which the bigram model has to be fit 
'''
def bigram_process(data):
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer(ngram_range=(1,2))
	vectorizer = vectorizer.fit(data)
	return vectorizer


'''
TFIDF_PROCESS takes the data to be fit as the input and returns a vectorizer of the tfidf as output 
Data - The data for which the bigram model has to be fit 
'''
def tfidf_process(data):
	from sklearn.feature_extraction.text import TfidfTransformer 
	transformer = TfidfTransformer()
	transformer = transformer.fit(data)
	return transformer


'''
RETRIEVE_DATA takes a CSV file as the input and returns the corresponding arrays of labels and data as output. 
Name - Name of the csv file 
Train - If train is True, both the data and labels are returned. Else only the data is returned 
'''
def retrieve_data(name="imdb_tr.csv", train=True):
	import pandas as pd 
	data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')
	X = data['text']
	
	if train:
		Y = data['polarity']
		return X, Y

	return X		

'''
STOCHASTIC_DESCENT applies Stochastic on the training data and returns the predicted labels 
Xtrain - Training Data
Ytrain - Training Labels
Xtest - Test Data 
'''
def stochastic_descent(Xtrain, Ytrain, Xtest):
	from sklearn.linear_model import SGDClassifier 
	clf = SGDClassifier(loss="hinge", penalty="l1", n_iter=20)
	print ("SGD Fitting")
	clf.fit(Xtrain, Ytrain)
	print ("SGD Predicting")
	Ytest = clf.predict(Xtest)
	return Ytest


'''
ACCURACY finds the accuracy in percentage given the training and test labels 
Ytrain - One set of labels 
Ytest - Other set of labels 
'''
def accuracy(Ytrain, Ytest):
	assert (len(Ytrain)==len(Ytest))
	num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
	n = len(Ytrain)  
	return (num*100)/n


'''
WRITE_TXT writes the given data to a text file 
Data - Data to be written to the text file 
Name - Name of the file 
'''
def write_txt(data, name):
	data = ''.join(str(word) for word in data)
	file = open(name, 'w')
	file.write(data)
	file.close()
	pass 


if __name__ == "__main__":
	import time
	start = time.time()
	# print ("Preprocessing the training_data--")
	# imdb_data_preprocess(train_path,outpath,"imdb_train.csv", True)
	print ("Preprocessing the testing_data--")
	imdb_data_preprocess(test_path,outpath,"imdb_test.csv", True)
	print ("Done with preprocessing. Now, will retreieve the training data in the required format")

	# [Xtrain_text, Ytrain] = retrieve_data()
	# print ("Retrieved the training data. Now will retrieve the test data in the required format")
	# Xtest_text = retrieve_data(name=test_path, train=False)
	# print ("Retrieved the test data. Now will initialize the model \n\n")


	# print ("-----------------------ANALYSIS ON THE INSAMPLE DATA (TRAINING DATA)---------------------------")
	# uni_vectorizer = unigram_process(Xtrain_text)
	# print ("Fitting the unigram model")
	# Xtrain_uni = uni_vectorizer.transform(Xtrain_text)
	# print ("After fitting ")
	# #print ("Applying the stochastic descent")
	# #Y_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtrain_uni)
	# #print ("Done with  stochastic descent")
	# #print ("Accuracy for the Unigram Model is ", accuracy(Ytrain, Y_uni))
	# print ("\n")

	# bi_vectorizer = bigram_process(Xtrain_text)
	# print ("Fitting the bigram model")
	# Xtrain_bi = bi_vectorizer.transform(Xtrain_text)
	# print ("After fitting ")
	# #print ("Applying the stochastic descent")
	# #Y_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtrain_bi)
	# #print ("Done with  stochastic descent")
	# #print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))
	# print ("\n")

	# uni_tfidf_transformer = tfidf_process(Xtrain_uni)
	# print ("Fitting the tfidf for unigram model")
	# Xtrain_tf_uni = uni_tfidf_transformer.transform(Xtrain_uni)
	# print ("After fitting TFIDF")
	# #print ("Applying the stochastic descent")
	# #Y_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
	# #print ("Done with  stochastic descent")
	# #print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
	# print ("\n")


	# bi_tfidf_transformer = tfidf_process(Xtrain_bi)
	# print ("Fitting the tfidf for bigram model")
	# Xtrain_tf_bi = bi_tfidf_transformer.transform(Xtrain_bi)
	# print ("After fitting TFIDF")
	# #print ("Applying the stochastic descent")
	# #Y_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
	# #print ("Done with  stochastic descent")
	# #print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
	# print ("\n")


	# print ("-----------------------ANALYSIS ON THE TEST DATA ---------------------------")
	# print ("Unigram Model on the Test Data--")
	# Xtest_uni = uni_vectorizer.transform(Xtest_text)
	# print ("Applying the stochastic descent")
	# Ytest_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtest_uni)
	# write_txt(Ytest_uni, name="unigram.output.txt")
	# print ("Done with  stochastic descent")
	# print ("\n")


	# print ("Bigram Model on the Test Data--")
	# Xtest_bi = bi_vectorizer.transform(Xtest_text)
	# print ("Applying the stochastic descent")
	# Ytest_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtest_bi)
	# write_txt(Ytest_bi, name="bigram.output.txt")
	# print ("Done with  stochastic descent")
	# print ("\n")

	# print ("Unigram TF Model on the Test Data--")
	# Xtest_tf_uni = uni_tfidf_transformer.transform(Xtest_uni)
	# print ("Applying the stochastic descent")
	# Ytest_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
	# write_txt(Ytest_tf_uni, name="unigramtfidf.output.txt")
	# print ("Done with  stochastic descent")
	# print ("\n")

	# print ("Bigram TF Model on the Test Data--")
	# Xtest_tf_bi = bi_tfidf_transformer.transform(Xtest_bi)
	# print ("Applying the stochastic descent")
	# Ytest_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
	# write_txt(Ytest_tf_bi, name="bigramtfidf.output.txt")
	# print ("Done with  stochastic descent")
	# print ("\n")

	print ("Total time taken is ", time.time()-start, " seconds")
	pass
