import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
from itertools import combinations
import copy
from sklearn.preprocessing import StandardScaler


class NonLinCTAextSampleOrd():
    """
    Attributes:
    - features: List of DataFrames containing features.
    - targets_df: DataFrame containing target values.
    - clusters: Dictionary mapping cluster IDs to a list of elements.
    - neighbors: List of neighbor pairs.
    - neighbor_strengths: Dictionary storing strengths for each neighbor pair.
    - active_neighbor_strengths: Dictionary storing strengths only for pairs that could be aggregated in the next iteration (to optimize code)
    """
        
    def __init__(self, features, targets_df, neighbors=None):

        self.features = copy.deepcopy(features)
        # Features preprocessing
        for i in range(len(self.features)):
            scaler = StandardScaler() #it deals with nan values
            features[i] = pd.DataFrame(scaler.fit_transform(features[i]), columns=features[i].columns)

        if type(targets_df)==str:
            pd.read_csv(targets_df)
        else: self.targets_df = targets_df.copy(deep=True)

        # Initialize clusters: each element as a cluster
        self.clusters = {subid:[subid] for subid in targets_df.columns}

        # If neighbors are not provided, generate all possible neighbor pairs
        if neighbors:
            self.neighbors = copy.deepcopy(neighbors)
        else:
            self.neighbors = set(combinations(list(self.targets_df.columns), 2))

        # Get strengths of each element
        self.neighbor_strengths = self.initialize_neighbor_strengths()
        self.active_neighbor_strengths = copy.deepcopy(self.neighbor_strengths)


    def get_clusters(self):
        return self.clusters


    def compute_strength(self, cluster1, cluster2):
        """
        Compute the strength between cluster1 and cluster2. 
        In this case we consider as strength the correlation between the two vectors.
        """
        cluster1_value = self.targets_df[self.clusters[cluster1]].mean(axis=1)
        cluster2_value = self.targets_df[self.clusters[cluster2]].mean(axis=1)

        # Create a mask to filter nan values
        cluster1_mask = ~np.isnan(cluster1_value)
        cluster2_mask = ~np.isnan(cluster2_value)
        mask = cluster1_mask & cluster2_mask
        
        strength = np.corrcoef(cluster1_value[mask], cluster2_value[mask])[0, 1]
        return strength


    def initialize_neighbor_strengths(self):
        neighbor_strengths = {}

        print("Computing initial neighbors strengths...")
        for neighbor_pair in tqdm(self.neighbors, miniters=int(len(self.neighbors)/10), maxinterval=60*10, position=0, smoothing=0.01):
            neighbor1, neighbor2 = neighbor_pair
            strength = self.compute_strength(neighbor1, neighbor2)
            neighbor_strengths[neighbor_pair] = strength

        print()
        return  neighbor_strengths 


    """
    def average_with_nan(self, vec1, vec2):
    averaged_vec = np.where(np.isnan(vec1), vec2, np.where(np.isnan(vec2), vec1, (vec1 + vec2) / 2))
    return averaged_vec
    """


    def prepare_features(self, cluster1, cluster2):       
        x1 = pd.concat([feature[self.clusters[cluster1]].mean(axis=1) for feature in self.features], axis=1)
        x2 = pd.concat([feature[self.clusters[cluster2]].mean(axis=1) for feature in self.features], axis=1)
        x1.columns = range(len(x1.columns))
        x2.columns = range(len(x2.columns))
        
        # Identify features with all NaN values in either x1 or x2
        nan_features_x1 = x1.columns[x1.isna().all()]
        nan_features_x2 = x2.columns[x2.isna().all()]
        nan_features_to_drop = set(nan_features_x1) | set(nan_features_x2)

        # Eliminate features with all NaN values in either x1 or x2
        x1 = x1.drop(columns=nan_features_to_drop)
        x2 = x2.drop(columns=nan_features_to_drop)

        x_aggr = pd.concat([feature[self.clusters[cluster1] + self.clusters[cluster2]].mean(axis=1) for feature in self.features], axis=1)
        x_aggr = x_aggr.drop(columns=nan_features_to_drop)

        return x1, x2, x_aggr 
    

    def prepare_target(self, cluster1, cluster2):
        y1 = self.targets_df[self.clusters[cluster1]].mean(axis=1)
        y2 = self.targets_df[self.clusters[cluster2]].mean(axis=1)
        y1 = y1 - np.nanmean(y1)
        y2 = y2 - np.nanmean(y2)

        y_aggr = self.targets_df[self.clusters[cluster1] + self.clusters[cluster2]].mean(axis=1)
        y_aggr = y_aggr - np.nanmean(y_aggr)
        return y1, y2, y_aggr


    def drop_observations_with_nan(self, x1, x2, x_aggr, y1, y2, y_aggr):
        # Drop rows with NaN values for each variable separately
        x1 = x1.dropna()
        x2 = x2.dropna()
        x_aggr = x_aggr.dropna()
        y1 = y1.dropna()
        y2 = y2.dropna()
        y_aggr = y_aggr.dropna()

        # Find the common indices across all variables
        common_indices = x1.index.intersection(x2.index).intersection(x_aggr.index).intersection(y1.index).intersection(y2.index).intersection(y_aggr.index)

        # Filter variables to keep only rows present in all variables
        x1 = x1.loc[common_indices].values
        x2 = x2.loc[common_indices].values
        x_aggr = x_aggr.loc[common_indices].values
        y1 = y1.loc[common_indices].values
        y2 = y2.loc[common_indices].values
        y_aggr = y_aggr.loc[common_indices].values

        return x1, x2, x_aggr, y1, y2, y_aggr


    def compute_VALscores(self, cluster1, cluster2):
        x1, x2, x_aggr = self.prepare_features(cluster1, cluster2)
        y1, y2, y_aggr = self.prepare_target(cluster1, cluster2)

        x1, x2, x_aggr, y1, y2, y_aggr = self.drop_observations_with_nan(x1, x2, x_aggr, y1, y2, y_aggr)
        
        # features "x" already standardized when initializing the class
        target1_regr = LinearRegression()
        target2_regr = LinearRegression()
        aggr_regr = LinearRegression()

        target1_regr.fit(x1,y1)
        target2_regr.fit(x2,y2)
        aggr_regr.fit(x_aggr,y_aggr)

        # we are now ready to perform the three linear regressions: the two individual ones and the one with aggregated targets
        # if for both it is convenient to aggregate, we do so 

        ### variance ###
        D = x1.shape[1] 
        n = x1.shape[0]
        preds1 = target1_regr.predict(x1)
        preds2 = target2_regr.predict(x2)
        preds_aggr = aggr_regr.predict(x_aggr)
        residuals1 = y1 - preds1
        residuals2 = y2 - preds2
        residuals_aggr = y_aggr - preds_aggr
        s_squared1 = np.dot(residuals1.reshape(1,n),residuals1)/(n-D-1)
        s_squared2 = np.dot(residuals2.reshape(1,n),residuals2)/(n-D-1)
        s_squared_aggr = np.dot(residuals_aggr.reshape(1,n),residuals_aggr)/(n-D-1)

        var1 = s_squared1*D/(n-1)
        var2 = s_squared2*D/(n-1)
        var_aggr = s_squared_aggr*D/(n-1)

        #aggr_r2 = cross_val_score(aggr_regr, x_aggr, y, cv=TimeSeriesSplit(-self.n_val), scoring='r2')
        #bivariate_r2 = cross_val_score(bivariate_regr, x, y, cv=TimeSeriesSplit(-self.n_val), scoring='r2')

        ### bias ### 
        r2_1 = r2_score(y1,preds1)
        r2_2 = r2_score(y2,preds2)
        r2_aggr = r2_score(y_aggr,preds_aggr)
        
        ### the following two are not needed but they can help to monitor the performances
        #r2_aggr_1 = r2_score(y1,preds_aggr)
        #r2_aggr_2 = r2_score(y2,preds_aggr)

        ### all equations of biases, not needed for the final threshold
        #bias1 = (np.var(y1,ddof=1)-s_squared1)*(1-r2_1)
        #bias2 = (np.var(y2,ddof=1)-s_squared2)*(1-r2_2)

        s_squaredF1 = (np.var(y1,ddof=1)-s_squared1)
        s_squaredF2 = (np.var(y2,ddof=1)-s_squared2)
        s_squaredFaggr = (np.var(y_aggr,ddof=1)-s_squared_aggr)

        #bias_aggr1 = s_squaredF1 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2))
        #bias_aggr2 = s_squaredF2 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF2*(r2_2) - s_squaredF1*(r2_1))

        ### these are the needed ones
        r2_1_weighted = r2_1*s_squaredF1
        r2_2_weighted = r2_2*s_squaredF2
        r2_aggr_weighted = r2_aggr*s_squaredFaggr

        #print(var1,var2,var_aggr,bias1,bias2,bias_aggr1,bias_aggr2)
        #print(s_squaredF1,s_squaredFaggr*r2_aggr, 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2)), s_squaredF2, 0.5*(s_squaredF2*(r2_2) - s_squaredF1*(r2_1)))

        #print(var1-var_aggr,var2-var_aggr,r2_aggr_weighted-0.5*r2_1_weighted-0.5*r2_2_weighted)
        #print(f'Basins: {cluster1, cluster2}, \nR2 coefficients: {r2_1,r2_2}, \naggregating: {r2_aggr,r2_aggr_1,r2_aggr_2}\n',flush=True)
        return var1,var2,var_aggr,r2_1_weighted,r2_2_weighted,r2_aggr_weighted


    def check_aggregation(self, cluster1, cluster2):
        var1,var2,var_aggr,r2_1_weighted,r2_2_weighted,r2_aggr_weighted = self.compute_VALscores(cluster1, cluster2)

        if (var1+r2_aggr_weighted>=var_aggr+0.5*(r2_1_weighted+r2_2_weighted)) & (var2+r2_aggr_weighted>=var_aggr+0.5*(r2_1_weighted+r2_2_weighted)): 
            return True
        else:
            return False
        

    def get_new_cluster_key(self):
        cluster_ID = 0
        new_key = f'cluster_{cluster_ID}'

        while new_key in self.clusters:
            cluster_ID += 1
            new_key = f'cluster_{cluster_ID}'
        
        return new_key
    

    def update_clusters(self, cluster1, cluster2):
        """
        Update clusters dictionary
        """
        if not cluster1.startswith('cluster') and not cluster2.startswith('cluster'):
            key = self.get_new_cluster_key()
            self.clusters[key] = [cluster1, cluster2]
            del self.clusters[cluster1]
            del self.clusters[cluster2]

        elif cluster1.startswith('cluster') and not cluster2.startswith('cluster'):    
            key = cluster1
            self.clusters[key].append(cluster2)
            del self.clusters[cluster2]

        elif not cluster1.startswith('cluster') and cluster2.startswith('cluster'):    
            key = cluster2
            self.clusters[key].append(cluster1)
            del self.clusters[cluster1]

        else: #cluster1.startswith('cluster') and cluster2.startswith('cluster'):    
            cluster1_ID = int(cluster1.split('_')[1]) 
            cluster2_ID = int(cluster2.split('_')[1]) 
            cluster_ID = min(cluster1_ID, cluster2_ID)
            del_cluster_ID = max(cluster1_ID, cluster2_ID)
            key = f'cluster_{cluster_ID}'
            del_key = f'cluster_{del_cluster_ID}'
            self.clusters[key].extend(self.clusters[del_key])
            del self.clusters[del_key]

        return key


    def update_neighbors(self, cluster1, cluster2, key):
        """
        Repopulate active_neighbor_strengths with pairs that had their strength updated 
        or simply update them if they are already present (same code for both cases)
        """

        # Delete neighbor relation between the merged clusters
        if (cluster1, cluster2) in self.neighbor_strengths:
            del self.neighbor_strengths[(cluster1, cluster2)]
            del self.active_neighbor_strengths[(cluster1, cluster2)]
        else:
            del self.neighbor_strengths[(cluster2, cluster1)]
            del self.active_neighbor_strengths[(cluster2, cluster1)]

        # Update neighbors strenghts
        for key_temp in list(self.neighbor_strengths.keys()):
            neighbor1, neighbor2 = key_temp

            if neighbor1 in (cluster1, cluster2) or neighbor2 in (cluster1, cluster2):
                if neighbor1 in (cluster1, cluster2):
                    neighbor_outside_cluster = neighbor2
                    new_key = (key, neighbor_outside_cluster)
                    inverse_new_key = (neighbor_outside_cluster, key)
                else:
                    neighbor_outside_cluster = neighbor1
                    new_key = (neighbor_outside_cluster, key)
                    inverse_new_key = (key, neighbor_outside_cluster)

                # Check if 'neighbor_inside_cluster' was previously a cluster,
                # and its index matches the index of the new cluster.
                # note: inverse_new_key != key_temp always
                if (new_key == key_temp): 
                    strength = self.compute_strength(neighbor_outside_cluster, key)
                    self.neighbor_strengths[key_temp] = strength 
                    self.active_neighbor_strengths[key_temp] = strength # whether key_temp was there or not
                else:
                    # Check that the same strength is not already updated (because of a common neighbor) 
                    # or that it won't be updated later (by entering condition (new_key == key_temp))
                    if inverse_new_key not in self.neighbor_strengths and new_key not in self.neighbor_strengths:
                        strength = self.compute_strength(neighbor_outside_cluster, key)
                        self.neighbor_strengths[new_key] = strength
                        self.active_neighbor_strengths[new_key] = strength

                    del self.neighbor_strengths[key_temp]
                    if key_temp in self.active_neighbor_strengths:
                        del self.active_neighbor_strengths[key_temp]



    def compute_clusters(self):
        # Get neighbor pairs sorted by their strengths  
        sorted_strengths = sorted(self.active_neighbor_strengths.keys(), key=self.active_neighbor_strengths.get, reverse=True)
        
        print("Computing clusters...")
        total = len(self.targets_df.columns)
        progress_bar = tqdm(total=total, miniters=int(total/100), maxinterval=60*10, position=0, smoothing=0.01)
        
        # Iterate until there are no pairs that can be merged into a cluster 
        while len(self.active_neighbor_strengths) != 0:
    
            for pair in sorted_strengths:
                (cluster1, cluster2) = pair

                # Check if the pair can be merged 
                if self.check_aggregation(cluster1, cluster2):                     
                    # Update clusters dictionary with the new cluster and return its key
                    cluster_key = self.update_clusters(cluster1, cluster2)
                    
                    # Update neighbor_strengths and active_neighbor_strengths dictionaries with new strengths and neighbors
                    self.update_neighbors(cluster1, cluster2, cluster_key) #eventually add new pairs to active_neighbor_strengths
                    
                    # Get sorted pairs only of clusters which could be merged
                    sorted_strengths = sorted(self.active_neighbor_strengths.keys(), key=self.active_neighbor_strengths.get, reverse=True)
                    
                    progress_bar.update()
                    break
                
                else: 
                    # Remove the pair from active_neighbor_strengths so that, if also its neighbors do not change,
                    # in the next iteration we don't need to consider it during sorting and to run check_aggregation  
                    del self.active_neighbor_strengths[pair]

            else:
                break

        progress_bar.close()




    #def compute_CVscores(self, column1_list, column2):
    #    x_aggr,x,y = self.prepare_data(self.df[column1_list].mean(axis=1).values,self.df[column2].values, len(column1_list))
    #    aggr_regr = LinearRegression()
    #    bivariate_regr = LinearRegression()
    #    aggr_r2 = cross_val_score(aggr_regr, x_aggr, y, cv=self.n_val, scoring='r2')
    #    bivariate_r2 = cross_val_score(bivariate_regr, x, y, cv=self.n_val, scoring='r2')
    #    return np.mean(aggr_r2),np.mean(bivariate_r2)

    #def compute_VALscores(self, column1_list, column2):
    #    x_aggr,x,y = self.prepare_data(self.df[column1_list].mean(axis=1).values,self.df[column2].values, len(column1_list))
    #    aggr_regr = LinearRegression()
    #    bivariate_regr = LinearRegression()
    #    aggr_r2 = cross_val_score(aggr_regr, x_aggr, y, cv=TimeSeriesSplit(-self.n_val), scoring='r2')
    #    bivariate_r2 = cross_val_score(bivariate_regr, x, y, cv=TimeSeriesSplit(-self.n_val), scoring='r2')
    #    return np.mean(aggr_r2),np.mean(bivariate_r2)



    

    