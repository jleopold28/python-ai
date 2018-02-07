import numpy as np
from sklearn import preprocessing

# define sample data
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
 		       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

print("INPUT DATA:")
print(input_data)

# Preprocessing techniques

## Binarization
# When we want to convert our numerical values into boolean values.
# x >  2.1 return 1
# x <= 2.1 return 0
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBINARIZED DATA:\n", data_binarized)

## Mean removal
# remove the mean from our feature vector, so that each feature is centered on 0
# to remove bias from features in our feature vector
# Print mean and std dev
print("\nMEAN REMOVAL on axis 0: add columns down where x=0, x1+x2+x3+x4/4 = mean")
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))
# Remove mean
data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))
#print("\nSCALED DATA:")
#print("X - mean / std dev = Y")
#print(data_scaled)

## Scaling
# set max value to 1 and all other values relative to this
# Min max scaling - set max in each column to 1
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMIN MAX SCALED DATA:\n", data_scaled_minmax)

## Normalization
# modify the values in the feature vector so that we can measure them on a common scale
# L1 Normalization = LEAST ABSOLUTE DEVIATIONS - makes sure the sum of abs val = 1 in each row.
#   more robust becuase it is resistant to outliers in the data
#   safelty and effectively ignore outliers during calculations
# L2 Normalization = LEAST SQUARES - makes sure the sum of squares is 1
#   used when outlieres are important
# Normalize Data
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)
