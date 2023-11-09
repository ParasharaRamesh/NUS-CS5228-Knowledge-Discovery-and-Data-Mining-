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
            # iterate over non zero indices
            for u, v in self.Z:
                # common term
                diff = 2 * (self.M[u, v] - np.dot(self.W[u], self.H[:,
                                                             v]))  # taking dot product along the k dimension (e.g. 100 in this case)

                # calculate gradients
                w_grad = learning_rate * ((diff * self.H[:, v]) - (2 * lambda_reg * self.W[u]))
                h_grad = learning_rate * ((diff * self.W[u]) - (2 * lambda_reg * self.H[:, v]))

                # update step
                self.W[u] += w_grad
                self.H[:, v] += h_grad

            ### Your code ends here #################################################################
            #########################################################################################           

            # Print loss every 10% of the iterations
            if verbose == True:
                if (it % (num_iter / 10) == 0):
                    print('Loss: {:.5f} \t {:.0f}%'.format(self.calc_loss(), (it / (num_iter / 100))))

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
    closeness_scores = {node: 0.0 for node in G.nodes}

    #########################################################################################
    ### Your code starts here ###############################################################
    for node in G.nodes:
        # only includes the distances to the reachable nodes
        distance_to_reachable_nodes_from_this = nx.shortest_path_length(G, source=node)

        # remove the current node
        distance_to_reachable_nodes_from_this.pop(node)

        # calculate closeness
        N = len(distance_to_reachable_nodes_from_this.keys())
        total_distances = sum(distance_to_reachable_nodes_from_this.values())
        closeness_scores[node] = N / total_distances

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

    # Initialize E
    E = np.ones(len(node_list))[:, np.newaxis] / len(node_list)

    # Initialize c by considering out degree
    c = np.array([G.out_degree(node, weight="weight") for node in node_list])[:, np.newaxis]

    ### Your code ends here #################################################################
    ######################################################################################### 

    # Run the power method: iterate until differences between steps converges
    num_iter = 0
    while num_iter < max_iter:
        num_iter += 1

        #########################################################################################
        ### Your code starts here ###############################################################  
        follow = alpha * np.matmul(M, c)
        random_jump = (1 - alpha) * E
        c_new = follow + random_jump

        # check for convergence
        has_c_converged = np.allclose(c, c_new, atol=eps)

        # update it
        c = c_new

        if has_c_converged:
            print("Reached convergence")
            break

        ### Your code ends here #################################################################
        #########################################################################################            

    c = c / np.sum(c)

    ## Return the results as a dictiory with the nodes as keys and the PageRank score as values
    return {node_list[k]: score for k, score in enumerate(c.squeeze())}