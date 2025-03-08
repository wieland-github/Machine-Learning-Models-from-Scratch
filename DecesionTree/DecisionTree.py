import numpy as np

class Node():

    def __init__(self, feature=None, threshold=None, left=None, 
                 right=None, info_gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecesionTree():
    def __init__(self, max_depth=2, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples

        def split_data(self, dataset, feature, treshold):

            left_data = []
            right_data = []

            for row in dataset:
                if row[feature] <= treshold:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_data = np.array(left_data)
            right_data = np.array(right_data)        

            return left_data, right_data
        

        def entropy(self, label_values):
            
            entropy = 0
            
            labels = np.unique(label_values)

            for label in labels:
                #Find the ex that have the given label
                ex_labels = label_values[label_values == label]

                #Ratio of Label in labels 
                pX = len(ex_labels) / len(label_values)

                #calculate entropy
                entropy = entropy - pl * np.log2(pX)

                return entropy
            
        def information_gain(self, parent, left, right):
            
            information_gain = 0

            parent_entropy = self.entropy(parent)
            entropy_left, entropy_right = self.entropy(left), self.entropy(right)
            
            weight_entropy_left, weight_entropy_right = len(left) / len(parent), len(right) / len(parent)

            weighted_childreen_entropy = weight_entropy_left * entropy_left + weight_entropy_right + entropy_right

            information_gain = parent_entropy - weighted_childreen_entropy

            return information_gain

        def best_split(self, dataset, num_features, num_samples):
            #dataset (ndarray): The dataset to split.
            #num_samples (int): The number of samples in the dataset.
            #num_features (int): The number of features in the dataset.

            