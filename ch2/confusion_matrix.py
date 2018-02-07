import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


## CONFUSION MATRIX
# Figure or table that is used to describe the performance of a classifier.
# It is usually extracted from a test dataset for which the groud truth is known
# We compare each class with every other class and see how many samples are misclassified.
# During the construction of the table, we come across serveral key metrics:

# True +: These are samples for which we predicted 1 as output and ground truth is 1
# True -: These are samples for which we predicted 0 as output and ground truth is 0

# False +: These are samples for which we predicted 1 as the output but ground truth is 0 (Type I errorr)
# False -: These are samples for which we predicted 0 as the output but ground truth is 1 (Type II error)

# Depending on the problem, we may have to optimize our algorithm to reduce the false positive or false negative rate
# EXample, in biometric identification system, it is very important to avoid false positives, becuase the wrong people might
# get access to sensitive information


# Define sample labels
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Create confusion matrix
confusion_mat = confusion_matrix(true_labels, pred_labels)

# Visualize confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5) ## distict classes (outcomes) = 5 distinct labels of 0,1,2,3,4
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()

# Classification report
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(true_labels, pred_labels, target_names=targets))


# precision = (true pos)/(tru pos + false pos)
# recall = (tru pos)/ (true pos + false neg)
# fscore = wighted harmonic mean of the precicion and recall, between best 1 and worst 0
# support = number of occurances of each class in y_true
