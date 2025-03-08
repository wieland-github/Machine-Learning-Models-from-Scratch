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

class DecisionTree:
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
            entropy = entropy - pX * np.log2(pX)

        return entropy
        
    def information_gain(self, parent, left, right):
        
        information_gain = 0

        #Calculate weights for Entropy
        weight_entropy_left, weight_entropy_right = len(left) / len(parent), len(right) / len(parent)

        #Calculate Entropys
        parent_entropy = self.entropy(parent)
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        weighted_childreen_entropy = weight_entropy_left * entropy_left + weight_entropy_right * entropy_right

        #Calculate Information Gain
        information_gain = parent_entropy - weighted_childreen_entropy

        return information_gain

    def best_split(self, dataset, num_features, num_samples):
        #dataset (ndarray): The dataset to split.
        #num_samples (int): The number of samples in the dataset.
        #num_features (int): The number of features in the dataset.
        best_split = {"gain" : -1, "feature" : None, "threshold" : None,
                      "left_data" : None, "right_data" : None}
        
        for feature_idx in range(num_features):
            feature_values = dataset[:, feature_idx]

            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_data, right_data = self.split_data(dataset, feature_idx, threshold)

                if len(left_data) and len(right_data):
                    # get y values of the parent and left, right nodes
                    y, left_y, right_y = dataset[:, -1], left_data[:, -1], right_data[:, -1]

                    # compute information gain based on the y values
                    information_gain = self.information_gain(y, left_y, right_y)

                    # update the best split if conditions are met
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["left_data"] = left_data
                        best_split["right_data"] = right_data
                        best_split["gain"] = information_gain

        return best_split
    

    def leaf_value(self, y_values):
        most_common_value = np.bincount(y_values.astype(int)).argmax()
        return most_common_value

        
    def build_tree(self, dataset, current_tree_depth):

        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape

        if current_tree_depth >= self.max_depth or n_samples < self.min_samples or len(np.unique(y)) == 1:

            best_split = self.best_split(dataset, n_features, n_samples)


            if best_split["gain"] != 0:
                left_node = self.build_tree(best_split["left_data"], current_tree_depth + 1)
                right_node = self.build_tree(best_split["right_data"], current_tree_depth + 1)

                return Node(best_split["feature"], best_split["threshold"], left_node, 
                            right_node, best_split["gain"])
            
        leaf_value = self.leaf_value(y)


        return Node(value=leaf_value)
    
    def fit(self, X, y):
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset, current_tree_depth=0)


    def predict(self, X):
        predictions = []

        for x in X:
            prediction = self.make_prediction(x, self.root)
            predictions.append(prediction)  

        return np.array(predictions)  



    def make_prediction(self, x, node):

        if node.value != None:
            return node.value
        else:
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)