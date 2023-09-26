import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def get_noise_dbscan(X, eps=0.0, min_samples=0):
    # Changed it to list
    core_point_indices, noise_point_indices = None, None

    #########################################################################################
    ### Your code starts here ###############################################################

    ### 2.1 a) Identify the indices of all core points

    # 1.Find the distance matrix between every point
    distances_i_j = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

    '''
    #2. Explanation: 
    * The distance matrix contains a shape of (N,N) i.e. (70,70)
    * By using np.sum across axis 1 for that condition of being in the radius of eps that gives us the no of points in its epsilon neighbourhood. I.e it gives us a a shape of (N,) or (70,)
    * By using the condition >= minsamples it gets converted into a boolean array where each of the N indices has a boolean value
    * Using the np.where gives us the indices where there is a True boolean value
    '''
    is_core_point = np.sum(distances_i_j <= eps, axis=1) >= min_samples
    core_point_indices = np.where(is_core_point)[0]

    # 3. To identify noise points , first find all the non core points
    is_non_core_point = ~is_core_point
    non_core_point_indices = np.where(is_non_core_point)[0]

    ### Your code ends here #################################################################
    #########################################################################################

    #########################################################################################
    ### Your code starts here ###############################################################

    ### 2.1 b) Identify the indices of all noise points ==> noise_point_indices

    # 1. From all the non-core points some of them may be border points and the rest are noise points, so lets filter out the border points
    is_non_core_point_a_noise_point = np.array(
        [
            np.intersect1d(
                np.where(distances_i_j[non_core_point_index] <= eps)[0],
                # find the epsilon neighbourhood of each non core point
                core_point_indices  # check if there is an intersection with the core points identified
            ).size == 0  # if there is no intersection then it is a noise point! as if there was it is a border point
            for non_core_point_index in non_core_point_indices
        ]
    )

    # 2. Get the noise point indices using the boolean mask calculated before
    noise_point_indices = non_core_point_indices[is_non_core_point_a_noise_point]

    ### Your code ends here #################################################################
    #########################################################################################

    return core_point_indices, noise_point_indices


if __name__ == '__main__':
    X = pd.read_csv('data/a2-dbscan-toy-dataset.txt', header=None, sep=' ').to_numpy()

    cp, np = get_noise_dbscan(X, eps=0.1, min_samples=10)
    print(f"# cp is {len(cp)} && #np is {len(np)}")
    print(f"cp is {cp}")
    print(f"np is {np}")

# %%
