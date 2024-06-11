import argparse
import glob
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path
from sklearn import preprocessing
from sklearn.utils import check_random_state
from scipy.stats import pearsonr
import multiprocessing
from joblib import Parallel, delayed
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from scipy.stats import pearsonr
import csv 
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import r2_score
import logging 
logging.basicConfig(level=logging.INFO)

class NonLinCTA():
    """
        Class which takes as input a dataframe (or path) of features, a dataframe with the corresponding targets, the error allowed and the number of cross-validation batches
        The method compute_clusters prints and returns the list of aggregated targets
    """
    def __init__(self, df, targets_df, eps, n_val, neigh, scale=0.1, verbose=False):
        if type(df)==str:
            pd.read_csv(df)
        else: self.df = df.copy(deep=True)

        self.x = self.df.values
        self.x = preprocessing.scale(self.x, with_mean=True, with_std=True)

        if type(targets_df)==str:
            pd.read_csv(targets_df)
        else: self.targets_df = targets_df.copy(deep=True)

        self.eps = eps
        self.n_val = n_val
        self.clusters = []
        self.scale = scale
        self.neigh = neigh
        self.verbose = verbose
    def print_header(self):
        print("Dataset: \n{}".format(self.df))

    def compute_corr(self, column1, column2):
        return pearsonr(self.df[column1],self.df[column2])[0]

    def prepare_target(self, y1, y2, l):
        #y1 = preprocessing.scale(y1, with_mean=True)
        #y2 = preprocessing.scale(y2, with_mean=True)
        y1 = y1-np.mean(y1)
        y2 = y2-np.mean(y2)
        #y = np.concatenate((y1.reshape(-1,1),y2.reshape(-1,1)),axis=1)
        y_aggr = ((y1*l+y2)/(l+1)).reshape(-1,1)
        #y_aggr = preprocessing.scale(y_aggr, with_mean=True)

        return y1,y2,y_aggr

    def compute_VALscores(self, column1_list, column2):
        y1,y2,y_aggr = self.prepare_target(self.targets_df[column1_list].mean(axis=1).values,self.targets_df[column2].values, len(column1_list))
        # features "x" already standardized when initializing the class
        aggr_regr = LinearRegression()
        target1_regr = LinearRegression()
        target2_regr = LinearRegression()
        aggr_regr.fit(self.x,y_aggr)
        target1_regr.fit(self.x,y1)
        target2_regr.fit(self.x,y2)
        # we are now ready to perform the three linear regressions: the two individual ones and the one with aggregated targets
        # if for both it is convenient to aggregate, we do so 

        ### variance ###
        D = self.df.shape[1]
        n = self.df.shape[0]
        preds1 = target1_regr.predict(self.x)
        preds2 = target2_regr.predict(self.x)
        preds_aggr = aggr_regr.predict(self.x)
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
        r2_aggr_1 = r2_score(y1,preds_aggr)
        r2_aggr_2 = r2_score(y2,preds_aggr)

        ### all equations of biases, not needed for the final threshold
        bias1 = (np.var(y1,ddof=1)-s_squared1)*(1-r2_1)
        bias2 = (np.var(y2,ddof=1)-s_squared2)*(1-r2_2)

        s_squaredF1 = (np.var(y1,ddof=1)-s_squared1)
        s_squaredF2 = (np.var(y2,ddof=1)-s_squared2)
        s_squaredFaggr = (np.var(y_aggr,ddof=1)-s_squared_aggr)

        bias_aggr1 = s_squaredF1 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2))
        bias_aggr2 = s_squaredF2 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF2*(r2_2) - s_squaredF1*(r2_1))

        ### these are the needed ones
        r2_1_weighted = r2_1*s_squaredF1
        r2_2_weighted = r2_2*s_squaredF2
        r2_aggr_weighted = r2_aggr*s_squaredFaggr

        #print(var1,var2,var_aggr,bias1,bias2,bias_aggr1,bias_aggr2)
        #print(s_squaredF1,s_squaredFaggr*r2_aggr, 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2)), s_squaredF2, 0.5*(s_squaredF2*(r2_2) - s_squaredF1*(r2_1)))

        if self.verbose is True:
            print(var1-var_aggr,var2-var_aggr,r2_aggr_weighted-0.5*r2_1_weighted-0.5*r2_2_weighted)
            #print(var1-var_aggr,var2-var_aggr,r2_aggr-0.5*r2_1-0.5*r2_2)

        if self.verbose is True:
            print(f'Basins: {column1_list,column2}, \nR2 coefficients: {r2_1,r2_2}, \naggregating: {r2_aggr,r2_aggr_1,r2_aggr_2}\n',flush=True)
        return var1,var2,var_aggr,r2_1_weighted,r2_2_weighted,r2_aggr_weighted

    def find_neighbors(self, actual_clust, cols): 
        neighs = []
        for datum in actual_clust: # put in x,y the location on the x and y axes 
            x = float(datum.split('_')[1])
            y = float(datum.split('_')[2])
            for c in cols:
                cx = float(c.split('_')[1])
                cy = float(c.split('_')[2])
                if ((abs(x-cx)<self.scale) & (abs(y-cy)<self.scale)): neighs.append(c) # for droughts self.scale=0.1
        return neighs

    def find_aggregation(self, actual_clust, cols):
        if self.neigh==1: neigh_names = self.find_neighbors(actual_clust, cols)
        else: neigh_names=cols
        for i in neigh_names:
            if self.n_val>0:
                var1,var2,var_aggr,r2_1_weighted,r2_2_weighted,r2_aggr_weighted = self.compute_CVscores(actual_clust, i)
            else: 
                var1,var2,var_aggr,r2_1_weighted,r2_2_weighted,r2_aggr_weighted = self.compute_VALscores(actual_clust, i)
            #print(r1,r2)
            if (var1+r2_aggr_weighted-var_aggr-0.5*(r2_1_weighted+r2_2_weighted)>=self.eps) & (var2+r2_aggr_weighted-var_aggr-0.5*(r2_1_weighted+r2_2_weighted)>=self.eps): return i
            #if (r2-r1<=self.eps): return i
        return ''

    def compute_target_clusters(self):
        output = []
        cols = self.targets_df.columns # all the target columns not yet assigned to a cluster
        actual_cluster = []

        while(len(cols)>0):

            if (actual_cluster == []):
                actual_col = cols[0] # take the first target
                actual_cluster.append(actual_col) # append that target to the actual cluster
                cols = cols[cols.values!=actual_col] # remove actual column from the ones not assigned yet

            col_to_aggr = self.find_aggregation(actual_cluster, cols)
            if col_to_aggr != '':
                actual_cluster.append(col_to_aggr)
                cols = cols[cols.values!=col_to_aggr]
            else:
                output.append(actual_cluster)
                actual_cluster = []
        if (len(actual_cluster)>0): output.append(actual_cluster)
        return output

    def compute_CVscores(self, column1_list, column2):
        y1,y2,y_aggr = self.prepare_target(self.targets_df[column1_list].mean(axis=1).values,self.targets_df[column2].values, len(column1_list))
        # features "x" already standardized when initializing the class
        aggr_regr = LinearRegression()
        target1_regr = LinearRegression()
        target2_regr = LinearRegression()
        #aggr_regr.fit(self.x,y_aggr)
        #target1_regr.fit(self.x,y1)
        #target2_regr.fit(self.x,y2)

        # we are now ready to perform the three linear regressions: the two individual ones and the one with aggregated targets
        # if for both it is convenient to aggregate, we do so 

        ### variance ###
        D = self.df.shape[1]
        n = self.df.shape[0]
        preds1 = cross_val_predict(target1_regr, self.x, y1, cv=KFold(self.n_val, shuffle=True), method='predict')
        preds2 = cross_val_predict(target2_regr, self.x, y2, cv=KFold(self.n_val, shuffle=True), method='predict')
        preds_aggr = cross_val_predict(aggr_regr, self.x, y_aggr, cv=KFold(self.n_val, shuffle=True), method='predict')
        residuals1 = y1 - preds1
        residuals2 = y2 - preds2
        residuals_aggr = y_aggr - preds_aggr
        s_squared1 = np.dot(residuals1.reshape(1,n),residuals1)/(n-D-1)
        s_squared2 = np.dot(residuals2.reshape(1,n),residuals2)/(n-D-1)
        s_squared_aggr = np.dot(residuals_aggr.reshape(1,n),residuals_aggr)/(n-D-1)

        var1 = s_squared1*D/(n-1)
        var2 = s_squared2*D/(n-1)
        var_aggr = s_squared_aggr*D/(n-1)

        ### bias ### 
        r2_1 = r2_score(y1,preds1) 
        r2_2 = r2_score(y2,preds2)
        r2_aggr = r2_score(y_aggr,preds_aggr)
        ### the following two are not needed but they can help to monitor the performances
        r2_aggr_1 = r2_score(y1,preds_aggr)
        r2_aggr_2 = r2_score(y2,preds_aggr)

        ### all equations of biases, not needed for the final threshold
        bias1 = (np.var(y1,ddof=1)-s_squared1)*(1-r2_1)
        bias2 = (np.var(y2,ddof=1)-s_squared2)*(1-r2_2)

        s_squaredF1 = (np.var(y1,ddof=1)-s_squared1)
        s_squaredF2 = (np.var(y2,ddof=1)-s_squared2)
        s_squaredFaggr = (np.var(y_aggr,ddof=1)-s_squared_aggr)

        bias_aggr1 = s_squaredF1 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2))
        bias_aggr2 = s_squaredF2 - s_squaredFaggr*r2_aggr - 0.5*(s_squaredF2*(r2_2) - s_squaredF1*(r2_1))

        ### these are the needed ones
        r2_1_weighted = r2_1*s_squaredF1
        r2_2_weighted = r2_2*s_squaredF2
        r2_aggr_weighted = r2_aggr*s_squaredFaggr

        #print(var1,var2,var_aggr,bias1,bias2,bias_aggr1,bias_aggr2)
        #print(s_squaredF1,s_squaredFaggr*r2_aggr, 0.5*(s_squaredF1*(r2_1) - s_squaredF2*(r2_2)), s_squaredF2, 0.5*(s_squaredF2*(r2_2) - s_squaredF1*(r2_1)))

        if self.verbose is True:
            print(var1-var_aggr,var2-var_aggr,r2_aggr_weighted-0.5*r2_1_weighted-0.5*r2_2_weighted)
            #print(var1-var_aggr,var2-var_aggr,r2_aggr-0.5*r2_1-0.5*r2_2)

        if self.verbose is True:
            print(f'Basins: {column1_list,column2}, \nR2 coefficients: {r2_1,r2_2}, \naggregating: {r2_aggr,r2_aggr_1,r2_aggr_2}\n',flush=True)
        return var1,var2,var_aggr,r2_1_weighted,r2_2_weighted,r2_aggr_weighted    
    
class NonLinCFA_new():
    """
        Class which takes as input a dataframe (or path) of features, a dataframe with the corresponding targets, the error allowed and the number of cross-validation batches
        The method compute_clusters prints and returns the list of aggregated targets
    """
    def __init__(self, df, target, eps, n_val, neigh, scale=0.1, verbose=False):
        if type(df)==str:
            pd.read_csv(df)
        else: self.df = df.copy(deep=True)

        self.df = (self.df-self.df.mean())/self.df.std()

        if type(target)==str:
            pd.read_csv(target)
        else: self.target = target.copy(deep=True)

        self.eps = eps
        self.n_val = n_val
        self.clusters = []
        self.scale = scale
        self.neigh = neigh
        self.verbose = verbose
    def print_header(self):
        print("Dataset: \n{}".format(self.df))

    def compute_corr(self, column1, column2):
        return pearsonr(self.df[column1],self.df[column2])[0]

    def prepare_data(self, x1, x2, l):
        # 
        #x1 = preprocessing.scale(x1, with_mean=True)
        #x2 = preprocessing.scale(x2, with_mean=True)
        x = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1)),axis=1)
        x_aggr = ((x1*l+x2)/(l+1)).reshape(-1,1) 
        x_aggr = preprocessing.scale(x_aggr, with_mean=True)
        y = self.target.values
        y = preprocessing.scale(y, with_mean=True, with_std=True) 
        return x_aggr, x, y

    def compute_VALscores(self, column1_list, column2):
        logging.debug(f'Length of actual cluster: {self.df[column1_list].shape[1]}')
        logging.debug(f'Length of actual cluster: {len(column1_list)}')

        x_aggr,x_sep,y = self.prepare_data(self.df[column1_list].mean(axis=1).values,self.df[column2].values, len(column1_list))
        # features "x" already standardized when initializing the class

        aggr_regr = LinearRegression()
        sep_regr = LinearRegression()


        logging.debug(f'Length of remaining features: {len(self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))].columns)}')
        logging.debug(f'Length of remaining features + aggregated: {len(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_aggr)), axis=1).columns)}')
        logging.debug(f'Length of remaining features + individual: {len(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_sep)), axis=1).columns)}')
        
        x_aggr = pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_aggr)), axis=1)
        x_sep = pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_sep)), axis=1)
        x_aggr.columns = x_aggr.columns.astype(str)
        x_sep.columns = x_sep.columns.astype(str)
        aggr_regr.fit(x_aggr,y)
        sep_regr.fit(x_sep,y)

        r2_aggr = r2_score(y,aggr_regr.predict(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_aggr).add_suffix('aggr')), axis=1)))
        r2_sep = r2_score(y,sep_regr.predict(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_sep).add_suffix('sep')), axis=1)))
        return r2_sep,r2_aggr

    def find_neighbors(self, actual_clust, cols): 
        neighs = []
        for datum in actual_clust: # put in x,y the location on the x and y axes 
            x = float(datum.split('_')[1])
            y = float(datum.split('_')[2])
            for c in cols:
                cx = float(c.split('_')[1])
                cy = float(c.split('_')[2])
                if ((abs(x-cx)<self.scale) & (abs(y-cy)<self.scale)): neighs.append(c) # for droughts self.scale=0.1
        return neighs

    def find_aggregation(self, actual_clust, cols):
        if self.neigh==1: neigh_names = self.find_neighbors(actual_clust, cols)
        else: neigh_names=cols
        for i in neigh_names:
            if self.n_val>0:
                r2_sep, r2_aggr = self.compute_CVscores(actual_clust, i)
            else: 
                r2_sep, r2_aggr = self.compute_VALscores(actual_clust, i)
            #print(r1,r2)
            logging.info(f'R2 scores: {r2_sep},{r2_aggr}')
            if (r2_sep-r2_aggr <= self.eps): return i
            #if (r2-r1<=self.eps): return i
        return ''

    def compute_feature_clusters(self):
        output = []
        cols = self.df.columns # all the feature columns not yet assigned to a cluster
        actual_cluster = []
        logging.debug(cols)

        while(len(cols)>0):

            if (actual_cluster == []):
                actual_col = cols[0] # take the first feature
                actual_cluster.append(actual_col) # append that feature to the actual cluster
                cols = cols[cols.values!=actual_col] # remove actual column from the ones not assigned yet
                logging.debug(actual_cluster)
                logging.debug(cols)
            col_to_aggr = self.find_aggregation(actual_cluster, cols)
            
            if col_to_aggr != '':
                actual_cluster.append(col_to_aggr)
                cols = cols[cols.values!=col_to_aggr]
                logging.info('Adding one element')
            else:
                logging.info('Completed one cluster')
                output.append(actual_cluster)
                self.df['aggr_feat_'+str(len(cols))] = self.df[actual_cluster].mean(axis=1) 
                self.df = self.df.drop(actual_cluster, axis=1)
                if self.verbose is True:
                    print(self.df.columns.values)
                    print(self.df)
                actual_cluster = []
        if (len(actual_cluster)>0): output.append(actual_cluster)
        return output

    def compute_CVscores(self, column1_list, column2):
        logging.debug(f'Length of actual cluster: {self.df[column1_list].shape[1]}')
        logging.debug(f'Length of actual cluster: {len(column1_list)}')

        x_aggr,x_sep,y = self.prepare_data(self.df[column1_list].mean(axis=1).values,self.df[column2].values, len(column1_list))
        # features "x" already standardized when initializing the class

        aggr_regr = LinearRegression()
        sep_regr = LinearRegression()


        logging.debug(f'Length of remaining features: {len(self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))].columns)}')
        logging.debug(f'Length of remaining features + aggregated: {len(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_aggr)), axis=1).columns)}')
        logging.debug(f'Length of remaining features + individual: {len(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_sep)), axis=1).columns)}')
        
        #aggr_regr.fit(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_aggr)), axis=1),y)
        #sep_regr.fit(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_sep)), axis=1),y)
        input_aggr = pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_aggr)), axis=1)
        input_sep = pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_sep)), axis=1)
        input_aggr.columns = input_aggr.columns.astype(str)
        input_sep.columns = input_sep.columns.astype(str)

        r2_aggr = cross_val_score(aggr_regr, input_aggr, y, cv=self.n_val, scoring='r2')
        r2_sep = cross_val_score(sep_regr, input_sep, y, cv=self.n_val, scoring='r2')

        #r2_aggr = r2_score(y,aggr_regr.predict(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_aggr).add_suffix('aggr')), axis=1)))
        #r2_sep = r2_score(y,sep_regr.predict(pd.concat((self.df.loc[:,list(set(self.df.columns.values) - set(column1_list+[column2]))],pd.DataFrame(x_sep).add_suffix('sep')), axis=1)))
        return np.mean(r2_sep),np.mean(r2_aggr)



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

