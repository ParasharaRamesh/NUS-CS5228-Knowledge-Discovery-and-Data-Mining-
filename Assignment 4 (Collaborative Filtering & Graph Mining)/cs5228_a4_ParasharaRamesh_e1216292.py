import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths import *


    
##########################################################################
##
## Non-Negative Matrix Factorization
##
##########################################################################

class NMF:
    
    def __init__(self, M, k=100):
        self.M, self.k = M, k
    
        num_users, num_items = M.shape
        
        self.Z = np.argwhere(M != 0)
        self.W = np.random.rand(num_users, k)
        self.H = np.random.rand(k, num_items)

        
        
    def calc_loss(self):
        
        loss = np.sum(np.square((self.M - np.dot(self.W, self.H)))[self.M != 0])

        return loss    
    
    
    
    def fit(self, learning_rate=0.0001, lambda_reg=0.1, num_iter=100, verbose=False):
        for it in range(num_iter):

            #########################################################################################
            ### Your code starts here ############################################################### 


            
            
            ### Your code ends here #################################################################
            #########################################################################################           

            # Print loss every 10% of the iterations
            if verbose == True:
                if(it % (num_iter/10) == 0):
                    print('Loss: {:.5f} \t {:.0f}%'.format(self.calc_loss(), (it / (num_iter/100))))

        # Print final loss        
        if verbose == True:
            print('Loss: {:.5f} \t 100%'.format(self.calc_loss()))        
        
        
    def predict(self):
        #
        return np.dot(self.W, self.H)
    
    
    
    

    
##########################################################################
##
## Closeness Centrality
##
##########################################################################


def closeness(G):
    
    closeness_scores = { node:0.0 for node in G.nodes }
    
    #########################################################################################
    ### Your code starts here ############################################################### 
    

    
    
    ### Your code ends here #################################################################
    #########################################################################################         
        
    return closeness_scores
    
    
    
    
    
    
    
##########################################################################
##
## PageRank Centrality
##
##########################################################################


def create_transition_matrix(A):
   
    # Divide each value by the sum of its column
    # Matrix M is column stochastic
    M = A / (np.sum(A, axis=1).reshape(1, -1).T)
    
    # Set NaN value to 0 (default value of nan_to_num)
    # Required of the sum of a columns was 0 (if directed graph is not strongly connected)
    M = np.nan_to_num(M).T
    
    return np.asarray(M)



def pagerank(G, alpha=0.85, eps=1e-06, max_iter=1000):
   
    node_list = list(G.nodes())

    ## Convert NetworkX graph to adjacency matrix (numpy array)
    A = nx.to_numpy_array(G)
    
    ## Generate transition matrix from adjacency matrix A
    M = create_transition_matrix(A)

    
    E, c = None, None
    
    #########################################################################################
    ### Your code starts here ############################################################### 

    ## Initialize E and v
    

    
    ### Your code ends here #################################################################
    ######################################################################################### 

    # Run the power method: iterate until differences between steps converges
    num_iter = 0
    while True:
        
        num_iter += 1

        #########################################################################################
        ### Your code starts here ###############################################################  
        

        
        
        ### Your code ends here #################################################################
        #########################################################################################            
            
        pass

    c = c / np.sum(c)
        
    ## Return the results as a dictiory with the nodes as keys and the PageRank score as values
    return { node_list[k]:score for k, score in enumerate(c.squeeze()) }


    