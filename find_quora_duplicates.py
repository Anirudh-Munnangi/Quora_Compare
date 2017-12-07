# Library Imports
from __future__ import division
import json
import sys
import re
import math
from collections import Counter
import random

"""#GENERAL COMMENTS
#This program uses a logistic regression approach in order to perform the classification
#Four features are collected from the training data which are the common similarity measures in the literature
#They are as follows:
# 1) Jaccard Similarity
# 2) Cosine Similarity
# 3) Eucledian Distance
# 4) Dice Coefficient
# 5) Pearson Correlation Coefficient
#All the vectors are Tf-Idf weighted before finding the similarity. All the different words available in the training set are utilized in
# making the corpus dictionary of words to be utilized in the same.
#Author: Anirudh Munnangi, Graduate Student EECS, University of Cincinnati
"""

# VARIABLE DECLARATIONS(S)----------------------------------------------------------------------------------------------------------------------
question_dict={} # Dictionary containing question_key and corresponding question_text
words=[] #list containing all words
X=[]# Inputs
Y=[]# Outputs

#FUNCTION DECLARATION(S)------------------------------------------------------------------------------------------------------------------------

def Sim_func(str_a,str_b,Idf_dictionary): # Similarity values between two strings
	WORD = re.compile(r'\w+')
	words_a = WORD.findall(str_a)
	words_b = WORD.findall(str_b)
	dict_a=Counter(words_a)
	dict_b=Counter(words_b)
	returnval=[]
	intersection = set(dict_a.keys()) & set(dict_b.keys())
	for val in dict_a.keys():
		dict_a[val]=dict_a[val]*Idf_dictionary[val]
	for val in dict_b.keys():
		dict_b[val]=dict_b[val]*Idf_dictionary[val]

	#ADDING THE JACCARD DISTANCE
	numerator = sum([(dict_a[x]) * (dict_b[x]) for x in intersection])
	sum1 = sum([(dict_a[x])**2 for x in dict_a.keys()])
	sum2 = sum([(dict_b[x])**2 for x in dict_b.keys()])
	denominator = sum1+sum2-numerator
	if not denominator:
		returnval.append(0)
	else:
		returnval.append(float(numerator)/(denominator))

	#ADDING THE COSINE SIMILARITY DISTANCE
	denominator = math.sqrt(sum1) * math.sqrt(sum2)
	if not denominator:
		returnval.append(0)
	else:
		returnval.append(float(numerator)/(denominator))

	#ADDING THE EUCLEDIAN DISTANCE
	sum1=sum([(dict_a[x]-dict_b[x])**2 for x in intersection])
	returnval.append(math.sqrt(sum1))

	#ADDING DICE COEFFICIENT
	numerator = 2*sum([(dict_a[x]) * (dict_b[x]) for x in intersection])
	sum1 = sum([(dict_a[x])**2 for x in dict_a.keys()])
	sum2 = sum([(dict_b[x])**2 for x in dict_b.keys()])
	denominator=sum1+sum2
	if not denominator:
		returnval.append(0)
	else:
		returnval.append(float(numerator)/(denominator))

	#ADDING THE PEARSON CORRELATION COEFFICIENT
	ta=sum([(dict_a[x]) for x in intersection])
	tb=sum([(dict_b[x]) for x in intersection])
	m=len(intersection)
	numerator = m*(sum([(dict_a[x]) * (dict_b[x]) for x in intersection]))-ta*tb
	den=math.sqrt((m*(sum([(dict_a[x])**2 for x in intersection]))-ta**2)*(m*(sum([(dict_b[x])**2 for x in intersection]))-tb**2))
	if not denominator:
		returnval.append(0)
	else:
		returnval.append(float(numerator)/(denominator))
	return returnval


#INPUT FOR THE JSON DATA----------------------------------------------------------------------------------------------------------------------
def main():
	ips=int(raw_input())
	WORD = re.compile(r'\w+')
	if  not(ips>=1 and ips<=60000):
		print "Incorrect number of input lines \n"
		sys.exit()

	for _ in range(0,ips):
		ip=(sys.stdin.readline())# For speed improvement
		val=json.loads(ip)
		qn_text=val["question_text"]
		question_dict[val["question_key"]]=qn_text.lower()
		words_in_question=Counter(WORD.findall(qn_text.lower()))
		for val in words_in_question.keys():
			words.append(val)
	# GENERATING IDF TO BE USED WHILE PREPARING THE SIMILARITY MEASURES---------------------------------------------------------------------------

	# All the different words in the test data set to form a dictionary
	dict_all_words=Counter(words) # Dictionary containing all the words along with their counts
	for val in dict_all_words.keys():
		dict_all_words[val]=1/dict_all_words[val]# IDF

	# INPUT FOR ALL THE QUESTION KEY TRAINING PAIRS------------------------------------------------------------------------------------------------

	ips=int(raw_input())

	if  not(ips>=1 and ips<=25000):
		print "Incorrect number of inputs"
		sys.exit()

	for _ in range(0,ips):
		#ip=(raw_input()).split()
		ip=(sys.stdin.readline()).split()
		distance=Sim_func(question_dict[ip[0]],question_dict[ip[1]],dict_all_words)
		X.append(distance)
		if int(ip[2])==1:
			Y.append(1)
		else:
			Y.append(0)

	total_ips=len(X)

	# TESTING--------------------------------------------------------------------------------------------------------------------------------------

	tests=int(raw_input())

	if  not(tests>=1 and tests<=3000):
		print "Incorrect number of inputs"
		sys.exit()

	# FITTING THE LOGISTIC REGRESSION MODEL
	searchlist=range(0,total_ips)
	features=5
	weights=[random.random() for _ in range(0,features+1)]
	alpha=0.02# Learning Rate
	for _ in range(0,50):
		# Shuffling Dataset
		random.shuffle(searchlist)
		# Training
		for val in searchlist:
			ips=[1]+X[val]
			ops=Y[val]
			pro=sum([ips[ctr]*weights[ctr] for ctr in range(0,features+1)])
			lop=1/(1+math.exp(-1*pro))
			err=ops-lop
			# Non-Regularized Updates
			weights[0]=weights[0]+alpha*err*ips[0]
			weights[1]=weights[1]+alpha*err*ips[1]
			weights[2]=weights[2]+alpha*err*ips[2]
			weights[3]=weights[3]+alpha*err*ips[3]
			weights[4]=weights[4]+alpha*err*ips[4]
			weights[5]=weights[5]+alpha*err*ips[5]
	# Training done

	for _ in range(0,tests):
		#ip=(raw_input()).split()
		ip=(sys.stdin.readline()).split()
		distance=Sim_func(question_dict[ip[0]],question_dict[ip[1]],dict_all_words)
		ips=[1]+distance
		pro=sum([ips[ctr]*weights[ctr] for ctr in range(0,features+1)])
		lop=1/(1+math.exp(-1*pro))
		if lop>0.5:
			decision=1
		if lop<=0.5:
			decision=0
		line=ip[0]+" "+ip[1]+" "+str(decision)
		print line

if __name__=="__main__":
	main()
