# Tumor Classification
Random forest classifier achieves 96% accuracy on diagnosis from [Breast Cancer (Wisconsin) Data Set](https://www.kaggle.com/gargmanish/basic-machine-learning-with-cancer/data).

## Data
Data is taken from the [Breast Cancer (Wisconsin) Data Set](https://www.kaggle.com/gargmanish/basic-machine-learning-with-cancer/data).

## Example
`$ python tumor-classification.py`
![Breast Cancer (Wisconsin) Correlation Heatmap](https://github.com/SeanCooke/tumor-classification/blob/master/correlation-heatmap.png?raw=true)
~~~~
Percentage of training data that is benign:	0.63
Percentage of training data that is malignant:	0.37

Random Forest Classifier Accuracy: 0.96

Random Forest Classifier Confusion Matrix:
benign classified as benign:	106
malignant classified as benign:	2
benign classified as malignant:	4
malignant classified as malignant:	59

             precision    recall  f1-score   support

     benign       0.96      0.98      0.97       108
  malignant       0.97      0.94      0.95        63

avg / total       0.96      0.96      0.96       171

~~~~

## Results
This classifier assumes the true prevalence of malignant tumors is approximately equal to the sample prevalence of malignant tumors (37%).   70% of the [Breast Cancer (Wisconsin) Data Set](https://www.kaggle.com/gargmanish/basic-machine-learning-with-cancer/data) were used for training and 30% was used for testing.

Only 6 features `texture_mean`, `perimeter_mean`, `smoothness_mean`, `compactness_mean`, `symmetry_mean`, and `fractal_dimension_mean` were used to predict tumor diagnosis.  These features were chosen as they have a low correlation in the [correlation heatmap](https://github.com/SeanCooke/tumor-classification/blob/master/correlation-heatmap.png).

A random forest classifier correctly classifies the test data 96% of the time.  The precision and recall for both benign and malignant tumors are all above 94%.  In tumor classification, it is important to minimize the number of malignant tumors classified as benign tumors.  Our classifier has a false negative rate of 3%.

## References
* [Basic Machine Learning with Cancer](https://www.kaggle.com/gargmanish/basic-machine-learning-with-cancer/notebook)
* [Working With Text Data](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
