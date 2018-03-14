#!/usr/bin/env python

from __future__ import division
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
	data = pd.read_csv("data.csv", header=0)
	# features_mean will hold all mean columns
	features_mean =	list(data.columns[3:13])
	# binarizing diagnosis: malignant: 1, benign: 0
	data["diagnosis"] = data["diagnosis"].map({"M":1, "B":0})
	data.rename(columns={"diagnosis": "is_malignant"}, inplace=True)
	corr = data[features_mean].corr()
	sns.set(font_scale=0.45)
	sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f', xticklabels=features_mean, yticklabels=features_mean, cmap='coolwarm')
	plt.show()
	# only predict with columns with a low correlation
	prediction_columns = ["texture_mean", "perimeter_mean", "smoothness_mean", "compactness_mean", "symmetry_mean", "fractal_dimension_mean"]
	train, test = train_test_split(data, test_size = 0.3, random_state=42)
	train_X = train[prediction_columns]
	train_y = train.is_malignant
	test_X = test[prediction_columns]
	test_y = test.is_malignant
	probability_malignant = sum(train_y) / len(train_y)
	probability_benign = 1 - probability_malignant
	print "Percentage of training data that is benign:\t"+str(round(probability_benign, 2))
	print "Percentage of training data that is malignant:\t"+str(round(probability_malignant, 2))
	# random forest classifier
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(train_X, train_y)
	prediction = model.predict(test_X)
	# accuracy score
	accuracy_score = metrics.accuracy_score(prediction, test_y)
	print "\nRandom Forest Classifier Accuracy: "+str(round(accuracy_score, 2))
	# confusion matrix
	print "\nRandom Forest Classifier Confusion Matrix:"
	tn, fp, fn, tp = metrics.confusion_matrix(test_y, prediction).ravel()
	print "benign classified as benign:\t"+str(tn)
	print "malignant classified as benign:\t"+str(fp)
	print "benign classified as malignant:\t"+str(fn)
	print "malignant classified as malignant:\t"+str(tp)
	# classification report
	print "\n"+metrics.classification_report(test_y, prediction, target_names=["benign", "malignant"])
	false_negative_rate = fp/(tp+fp)
	# summary
	print "Summary:"
	print "This classifier assumes the true prevalence of malignant tumors is approximately equal to the sample prevalence of malignant tumors ("+str("%d" % round(probability_malignant*100.0))+"%).   70% of the Breast Cancer Wisconsin (Diagnostic) data were used for training and 30% was used for testing.  A random forest classifier correctly classifies the test data "+str("%d" % round(accuracy_score*100.0))+"% of the time.  In tumor classification, it is important to minimize the number of malignant tumors classified as benign tumors.  Our classifier has a false negative rate of "+str("%d" % round(false_negative_rate*100.0))+"%."

if __name__ == "__main__":
	main()
