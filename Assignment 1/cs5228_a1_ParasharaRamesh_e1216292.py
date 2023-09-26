import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import random

def clean(df_cars_dirty):
    """
    Handle all "dirty" records in the cars dataframe

    Inputs:
    - df_cars_dirty: Pandas dataframe of dataset containing "dirty" records

    Returns:
    - df_cars_cleaned: Pandas dataframe of dataset without "dirty" records
    """   
    
    # We first create a copy of the dataset and use this one to clean the data.
    df_cars_cleaned = df_cars_dirty.copy()

    #########################################################################################
    ### Your code starts here ###############################################################

    #filter points where manufacturing date is less than 2023
    df_cars_cleaned = df_cars_cleaned[df_cars_cleaned['manufactured'] <= 2023]

    #filter points where no_of_owners is >=1
    df_cars_cleaned = df_cars_cleaned[df_cars_cleaned["no_of_owners"] >= 1]

    #filter points where curb weight is only numeric and not string
    df_cars_cleaned = df_cars_cleaned[df_cars_cleaned["curb_weight"].apply(lambda x: x.isnumeric())]

    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_cars_cleaned



def handle_nan(df_cars_nan):
    """
    Handle all nan values in the cars dataframe

    Inputs:
    - df_cars_nan: Pandas dataframe of dataset containing nan values

    Returns:
    - df_cars_no_nan: Pandas dataframe of dataset without nan values
    """       

    # We first create a copy of the dataset and use this one to clean the data.
    df_cars_no_nan = df_cars_nan.copy()

    #########################################################################################
    ### Your code starts here ###############################################################

    #step1. remove NaN Url values
    df_cars_no_nan = df_cars_no_nan[df_cars_no_nan['url'].notna()]

    #step2. remove NaN make values
    df_cars_no_nan = df_cars_no_nan[df_cars_no_nan['make'].notna()]

    #step3a. compute the average value for price when grouped by (make & type of vehicle)
    df_cars_no_nan["price"] = df_cars_no_nan.groupby(['make', 'type_of_vehicle'])['price'].transform(lambda x: x.fillna(x.mean()))

    #step3b. since there are still 2 rows where the make & type of vehicle combo is unique, just find the average of the make column instead for these rows!
    for index, row in df_cars_no_nan.iterrows():
        if pd.isna(row['price']):
            make_value = row['make']
            average_value = df_cars_no_nan[df_cars_no_nan['make'] == make_value]['price'].mean()
            df_cars_no_nan.at[index, 'price'] = average_value

    #step4a. compute the average value for mileage when grouped by (make & type of vehicle)
    df_cars_no_nan["mileage"] = df_cars_no_nan.groupby(['make', 'type_of_vehicle'])['mileage'].transform(lambda x: x.fillna(x.mean()))

    #step4b. since there are still 2 rows where the make & type of vehicle combo is unique, just find the average of the make column instead for these rows!
    for index, row in df_cars_no_nan.iterrows():
        if pd.isna(row['mileage']):
            make_value = row['make']
            average_value = df_cars_no_nan[df_cars_no_nan['make'] == make_value]['mileage'].mean()
            df_cars_no_nan.at[index, 'mileage'] = average_value

    #step4c. since there is still 1 row where the make of the car is unique, we need to remove that row!
    df_cars_no_nan = df_cars_no_nan[df_cars_no_nan['mileage'].notna()]

    ### Your code ends here #################################################################
    #########################################################################################
    
    return df_cars_no_nan



def extract_facts(df_cars_facts):
    """
    Extract the facts as required from the cars dataset

    Inputs:
    - df_card_facts: Pandas dataframe of dataset containing the cars dataset

    Returns:
    - Nothing; you can simply us simple print statements that somehow show the result you
      put in the table; the format of the  outputs is not important; see example below.
    """       

    #########################################################################################
    ### Your code starts here ###############################################################

    #Q1
    print(f"1. Lowest price is {df_cars_facts['price'].min()} & Highest price is {df_cars_facts['price'].max()}")

    #Q2
    print(f"2. No of unique makes is {df_cars_facts['make'].nunique()}")

    #Q3
    sold_toyota_corolla_pre2010 = df_cars_facts[(df_cars_facts['make'] == "toyota") & (df_cars_facts['model'] == "corolla") & (df_cars_facts['manufactured'] < 2010)]
    print(f"3. No of sold toyota corolla before 2010 is {len(sold_toyota_corolla_pre2010)}")

    #Q4 top 3 most sold car makes, ( car make and no of sales? )
    df_cars_facts["sales"] = 1
    total_sales = df_cars_facts.groupby('make')['sales'].sum().reset_index()
    top_3_cars = total_sales.sort_values(by='sales', ascending=False).head(3)
    print(f"4. #1 sold car was {top_3_cars.iloc[0]['make']} and total sales was {top_3_cars.iloc[0]['sales']}")
    print(f"   #2 sold car was {top_3_cars.iloc[1]['make']} and total sales was {top_3_cars.iloc[1]['sales']}")
    print(f"   #3 sold car was {top_3_cars.iloc[2]['make']} and total sales was {top_3_cars.iloc[2]['sales']}")

    #Q5  Which SUV car model has been sold the most (give the model and the number of sales)?
    suv_cars = df_cars_facts[df_cars_facts['type_of_vehicle'] == "suv"]
    suv_cars["sales"] = 1
    model_total_sales = suv_cars.groupby('model')['sales'].sum().reset_index()
    most_sold_suv_model = model_total_sales.sort_values(by='sales', ascending=False).head(1)
    print(f"5. The model with most sales was '{most_sold_suv_model.iloc[0]['model']}' and total sales was {most_sold_suv_model.iloc[0]['sales']}")

    #Q6
    low_powered = df_cars_facts[df_cars_facts['power'] <= 60]
    low_powered_sales = low_powered.groupby('make')['price'].sum().reset_index()
    most_sold_low_powered = low_powered_sales.sort_values(by='price', ascending=False).head(1)
    print(f"6. Most sold low powered car make was {most_sold_low_powered.iloc[0]['make']} and the total sales were {most_sold_low_powered.iloc[0]['price']}")

    #Q7
    # need the make, model , year of manufacturing, power/engine ratio
    only_midsized_sedan = df_cars_facts[df_cars_facts['type_of_vehicle']=="mid-sized sedan"]
    only_midsized_sedan['power_to_engine'] = only_midsized_sedan['power']/only_midsized_sedan['engine_cap']
    max_ratio_index = only_midsized_sedan['power_to_engine'].idxmax()
    max_row = only_midsized_sedan.loc[max_ratio_index]
    print(f"7. Make={max_row['make']}, Model={max_row['model']}, YOM={max_row['manufactured']}, ratio={max_row['power_to_engine']}")
    
    #Q8
    #find pearson correlation between price & mileage | price & engine cap
    print(f"8. Pearson corr b/w price & mileage is {df_cars_facts['price'].corr(df_cars_facts['mileage'])} &")
    print(f" Pearson corr b/w price & engine_cap is {df_cars_facts['price'].corr(df_cars_facts['engine_cap'])}")


    ### Your code ends here #################################################################
    #########################################################################################




def kmeans_init(X, k, c1=None, method='kmeans++'):
    
    """
    Calculate the initial centroids for performing K-Means

    Inputs:
    - X: A numpy array of shape (N, F) containing N data samples with F features
    - k: number of centroids/clusters
    - c1: First centroid as the index of the data point in X
    - method: string that specifies the methods to calculate centroids ('kmeans++' or 'maxdist')

    Returns:
    - centroid_indices: NumPy array containing k centroids, represented by the
      indices of the respective data points in X
    """   
    
    centroid_indices = []
    
    # If the index of the first centroid index c1 is not specified, pick it randomly
    if c1 is None:
        c1 = np.random.randint(0, X.shape[0])
        
    # Add selected centroid index to list
    centroid_indices.append(c1)        
    
        
    # Calculate and add c2, c3, ..., ck 
    while len(centroid_indices) < k:
        c = None

        #########################################################################################
        ### Your code starts here ###############################################################

        all_distances = []

        #find distances to all centroids
        for centroid_index in centroid_indices:
            distances_to_curr_centroid = np.linalg.norm(X - np.array(X[centroid_index]), axis=1)
            # just ensure we only have the distances
            all_distances.append(distances_to_curr_centroid.tolist())

        #for each point in X find the distance to the closest cluster
        distances = [min(group) for group in zip(*all_distances)]

        if method == "kmeans++":
            #square each distance and divide by total to get probability
            square_distances = [distance**2 for distance in distances]
            denom = sum(square_distances)
            picking_probabilities = [sq_dist/denom for sq_dist in square_distances]

            #use random.choices to pick based on probability
            indices = list(range(len(X)))
            c = random.choices(indices, picking_probabilities)[0]

        elif method == "maxdist":
            c = distances.index(max(distances))

        ### Your code ends here #################################################################
        #########################################################################################                

        centroid_indices.append(c)
    
    # Return list of k centroid indices as numpy array (e.g. [0, 1, 2] for K=3)
    return np.array(centroid_indices)

#%%
