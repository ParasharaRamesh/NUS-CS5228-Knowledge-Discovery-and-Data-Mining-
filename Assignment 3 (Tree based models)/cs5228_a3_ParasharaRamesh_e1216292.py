import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DecisionStumpClassifier:
    
    def __init__(self):
        # Keeps the index of best feature and the value of the best threshold
        self.feature_idx, self.threshold = None, None
        # Keeps the indices of the data samples in the left and right child node
        self.y_left, self.y_right = None, None
        
        
    def calc_thresholds(self, x):
        """
        Calculates the set of all valid thresholds given a list of numerical values.
        The set of all valid thresholds is a set of minimum size that contains
        the values that would split the input list of values into two sublist:
        (1) all values less or equal the the threshold
        (2) all values larger than the threshold

        Inputs:
        - x: A numpy array of shape (N,) containing N numerical values, 
             Example: x = [4, 1, 2, 1, 1, 3]
             
        Returns:
        - Set of numerical values representing the thresholds 
          Example for input above: set([1.5, 2.5, 3.5])
        """              
        
        ## Get unique values to handle duplicates; return values will already be sorted
        values_sorted = np.unique(x)
        
        ## Return the "middle values" as thresholds
        return (values_sorted[:-1] + values_sorted[1:]) / 2.0
        
        
        
    def calc_gini_score_node(self, y):
        """
        Calculates Gini Score of a node in the Decision Tree

        Inputs:
        - y: A numpy array of shape (N,) containing N numerical values representing class labels, 
             Example: x = [0, 1, 1, 0, 0, 0, 2]
             
        Returns:
        - Gini Score of node as numeriv value
        """            
        
        gini = None
        
        ################################################################################
        ### Your code starts here ######################################################


        
        ### Your code ends here ########################################################
        ################################################################################        
        
        return gini
    
        
    def calc_gini_score_split(self, y_left, y_right):
        """
        Calculates Gini Score of a split; since we only consider binary splits, 
        this is the weighted average of the Gini Score for both child nodes.

        Inputs:
        - y_left:  A numpy array of shape (N,) containing N numerical values representing class labels, 
                   Example: x = [0, 1, 1, 0, 0, 0, 2]
        - y_right: A numpy array of shape (N,) containing N numerical values representing class labels, 
                   Example: x = [1, 2, 2, 2, 0, 2]
             
        Returns:
        - Gini Score of split as numeric value
        """   
        
        
        split_score = None
        
        ################################################################################
        ### Your code starts here ######################################################
        
        

        ### Your code ends here ########################################################
        ################################################################################        
        
        return split_score
        
        
    def fit(self, X, y):
        """
        Trains the Decision Stump, i.e., finds the best split w.r.t. all features
        and all possible thresholds

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
        - y: A numpy array of shape (N,) containing N numerical values representing class labels
             
        Returns:
        - self
        """           
        
        # Initilize the score with infinity
        score = np.inf
        
        ## Loop through all features (columns of X) to find the best split
        for feature_idx in range(X.shape[1]):
            
            # Get all values for current feature
            values = X[:, feature_idx]
            
            # Loop over all possible threshold; we are keeping it very simple here
            # all possible thresholds (see method above)
            for threshold in self.calc_thresholds(values):
                ################################################################################
                ### Your code starts here ######################################################
                pass



                ### Your code ends here ########################################################
                ################################################################################

        ## Return DecisionStumpClassifier object
        return self                 
                
                    
    def predict(self, X):
        """
        Uses Decision Stump to predict the class labels for a set of data points

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
             
        Returns:
        - y_pred: A numpy array of shape (N,) containing N integer values representing the predicted class labels
        """
                
        y_pred = np.zeros((X.shape[0], ))
        
        ################################################################################
        ### Your code starts here ######################################################
        
        
        
            
        ### Your code ends here ########################################################
        ################################################################################            
                    
        return y_pred    
    
    
    
    
    
class AdaBoostTreeClassifier:
    
    def __init__(self, n_estimators=50):
        self.estimators, self.alphas = [], []
        self.n_estimators = n_estimators
        self.classes = None
        
    
    def fit(self, X, y):
        """
        Trains the AdaBoost classifier using Decision Trees as Weak Learners.

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
        - y: A numpy array of shape (N,) containing N numerical values representing class labels
             
        Returns:
        - self
        """        
        
        N = X.shape[0]
        # Initialize the first sample as the input
        D, d = X, y
        # Initialize the sample weights uniformly
        w = np.full((N,), 1/N)
        # Create the list of class labels from all unique values in y
        self.classes = np.unique(y)
        
        for m in range(self.n_estimators):

            # Step 1: Train Weak Learner on current datset sample
            estimator = DecisionStumpClassifier().fit(D, d)
            
            # If you don't trust your implementation of DecisionStumpClassifier or just for testing,
            # you can use the line below to work with the implementation of sklearn instead
            #estimator = DecisionTreeClassifier(max_depth=1).fit(D, d)
            
            # Add current stump to sequences of all Weak Learners
            self.estimators.append(estimator)
            
            ################################################################################
            ### Your code starts here ######################################################
            
            # We give some guides but feel free to ignore if you have a better solution
            # misclassified, e, a, w below are all assumed to be numpy arrays
            
            # Step 2: Identify all samples in X that get misclassified with the current estimator
            misclassified = None

        
            # Step 3: Calculate the total error for current estimator
            e = None

            
            # Step 4: Calculate amount of say for current estimator and keep track of it
            # (we need the amounts of say later for the predictions)
            a = None

            
            # Step 3: Update the sample weights w based on amount of say a
            w = None


            # Step 4: Sample next D and d
            D, d = None, None
            
            ### Your code ends here ########################################################
            ################################################################################            
            
        # Convert the amounts-of-say to a numpy array for convenience
        # We need this later for making our predictions
        self.alphas = np.array(self.alphas)
        
        ## Return AdaBoostTreeClassifier object
        return self         
        

            
    def predict(self, X):
        """
        Predicts the class label for an array of data points

        Inputs:
        - X: A numpy array of shape (N, D) containing N data samples presented by D features,
             
        Returns:
        - y_pred: A numpy array of shape (N,) containing N integer values representing the predicted class labels
        """        
        
        # Predict the class labels for each sample individually
        return np.array([ self.predict_sample(x) for x in X ])
        
        
    def predict_sample(self, x):
        """
        Predicts the class label for a single data point

        Inputs:
        - x: A numpy array of shape (D, ) containing D features,
             
        Returns:
        - y_pred: integer value representing the predicted class label
        """        
        
        y = None
        
        # The predict method of our classifier expects a matrix,
        # so we need to convert our sample to a ()
        x = np.array([x])
        
        # Create a vector for all data points and all n_estimator predictions
        y_estimators = np.full(self.n_estimators, -1, dtype=np.int16)
        
        # Stores the score for each class label, e.g.,
        # class_scores[0] = class score for class 0
        # class_scores[1] = class score for class 1
        # ...
        class_scores = np.zeros(len(self.classes))
        
        y_pred = None
        
        ################################################################################
        ### Your code starts here ######################################################        
        

        
        ### Your code ends here ########################################################
        ################################################################################
        
        return y_pred    