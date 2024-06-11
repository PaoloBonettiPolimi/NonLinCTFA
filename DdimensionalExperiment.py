import logging
logging.basicConfig(#filename='/Users/paolo/Documents/MultiLinCFA/synthExp_varySamples.log',
                    #filemode='a',
                    level=logging.WARNING)
import sys
sys.path.append("../MultiLinCFA/MultiLinCFA")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.decomposition import PCA
import argparse
from random import randrange
import pandas as pd
from MultiLinCFA import NonLinCTA,NonLinCFA_new
from MultiLinCFA_extended import NonLinCTAext
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import random
import datetime

def generate_n_targets(df, noise=1, n_targets=10, seed=0):
    coeffs=[]
    n_data = df.shape[0]
    
    random.seed(seed)
    np.random.seed(seed)
    y = np.zeros((n_data,n_targets))
    
    ### standardize features dataframe
    scaler = preprocessing.StandardScaler().fit(df)
    x = scaler.transform(df)
    df = pd.DataFrame(x,columns=df.columns)
    
    for i in range(n_targets):
        dummy = bool(random.getrandbits(1))
        if dummy is True:
            coeffs.append(np.random.uniform(low=-1,high=-0.5,size=df.shape[1]))
        else:
            coeffs.append(np.random.uniform(low=0.5,high=1,size=df.shape[1]))

        epsilon = np.random.normal(0, noise, size=(df.shape[0],1))
        y[:,i] = np.dot(x,coeffs[-1]).reshape(-1,) + epsilon.reshape(-1,)
        
    return y,coeffs

def generate_correlated_features(n_feat, train_dim, test_dim, seed, n_tasks, noise):
    x = np.zeros((train_dim+test_dim,n_feat))
    
    x[:,0] = np.random.uniform(size=train_dim+test_dim)
    
    for i in range(n_feat-1):
        j = randrange(i+1)
        x[:,i+1] = 0.7*np.random.uniform(size=train_dim+test_dim) + 0.3*x[:,j]
    
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)

    df = pd.DataFrame(x) 
    df = df.sample(n_feat,axis=1,random_state=0)
    df = df.sample(train_dim+test_dim,axis=0,random_state=seed).reset_index(drop=True)
    df_train = df.loc[:train_dim-1,:]
    df_test = df.loc[train_dim:,:]

    y,coeffs = generate_n_targets(df, noise=noise, n_targets=n_tasks, seed=seed)
    y_df = pd.DataFrame(y)
    y_df_train = y_df.loc[:train_dim-1,:]
    y_df_test = y_df.loc[train_dim:,:]
    return df_train,df_test,y_df_train,y_df_test

def generate_features_from_droughts(n_feat, train_dim, test_dim, seed, n_tasks, noise):

    if 1==0:
        basins = ['Adda','Ticino','Lambro_Olona','Piemonte_Nord','Piemonte_Sud','Garda_Mincio','Oglio_Iseo','Emiliani2','Emiliani1','Dora']

        df_test = pd.read_csv("/Users/paolo/Documents/MultiLinCFA/droughts/NonLinCFA_final_features/temp_prec/Adda_nonLinCFA_CMI_test.csv")
        df_train = pd.read_csv("/Users/paolo/Documents/MultiLinCFA/droughts/NonLinCFA_final_features/temp_prec/Adda_nonLinCFA_CMI_train.csv")
        df_val = pd.read_csv("/Users/paolo/Documents/MultiLinCFA/droughts/NonLinCFA_final_features/temp_prec/Adda_nonLinCFA_CMI_val.csv")
        df = pd.concat((df_train,df_val,df_test),axis=0).reset_index(drop=True)

        for basin in basins[1:]:
            df_test = pd.read_csv("/Users/paolo/Documents/MultiLinCFA/droughts/NonLinCFA_final_features/temp_prec/"+basin+"_nonLinCFA_CMI_test.csv")
            df_train = pd.read_csv("/Users/paolo/Documents/MultiLinCFA/droughts/NonLinCFA_final_features/temp_prec/"+basin+"_nonLinCFA_CMI_train.csv")
            df_val = pd.read_csv("/Users/paolo/Documents/MultiLinCFA/droughts/NonLinCFA_final_features/temp_prec/"+basin+"_nonLinCFA_CMI_val.csv")
            df_curr = pd.concat((df_train,df_val,df_test),axis=0).reset_index(drop=True)
            df_curr = df_curr.add_suffix(basin)
            df = pd.concat((df,df_curr),axis=1)

    else:
        df = pd.read_csv("../MultiLinCFA/droughts/generatedData.csv")

    df_cube = np.power(df,3).add_suffix('_cube')
    df = pd.concat((df,df_cube),axis=1)

    df = pd.read_csv("/Users/paolo/Documents/MultiLinCFA/full_test.csv")
    
    df = df.sample(n_feat,axis=1,random_state=0)
    df = df.sample(train_dim+test_dim,axis=0,random_state=seed).reset_index(drop=True)
    df_train = df.loc[:train_dim-1,:]
    df_test = df.loc[train_dim:,:]

    y,coeffs = generate_n_targets(df, noise=noise, n_targets=n_tasks, seed=seed)
    y_df = pd.DataFrame(y)
    y_df_train = y_df.loc[:train_dim-1,:]
    y_df_test = y_df.loc[train_dim:,:]
    return df_train,df_test,y_df_train,y_df_test

### print 95% CI considering the distribution to be gaussian
def print_95CI(mylist):
    return str(round(np.mean(mylist),6))+'±'+str(round(1.96*np.std(mylist)/np.sqrt(len(mylist)),6))

### compute 95% CI considering the distribution to be gaussian
def compute_95CI(mylist):
    #return str(round(np.mean(mylist),6))+'±'+str(round(1.96*np.std(mylist)/np.sqrt(3000),6))
    return np.mean(mylist)-1.96*np.std(mylist)/np.sqrt(3000),np.mean(mylist)+1.96*np.std(mylist)/np.sqrt(len(mylist))

### main run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", default=10, type=float)
    parser.add_argument("--n_reps", default=1, type=int)
    parser.add_argument("--n_tasks", default=10, type=int)
    parser.add_argument("--n_feats", default=50, type=int)
    parser.add_argument("--train_dim", default=100, type=int)
    parser.add_argument("--eps1", default=0, type=float)
    parser.add_argument("--eps2", default=0.0001, type=float)

    args = parser.parse_args()
    print(args)
    logging.warning(f"noise: {args.noise}, n_reps: {args.n_reps}, n_tasks: {args.n_tasks}, n_feats: {args.n_feats}, train_dim: {args.train_dim}, eps: {args.eps1,args.eps2}")
    
    logging.warning(f"Current time: {datetime.datetime.now()}")

    ##################### experiments ########################

    ### basic configuration
    r2_single = []
    r2_aggr = []
    r2_aggr_both = []
    mse_single = []
    mse_aggr = []
    mse_aggr_both = []
    r2_relIncr = []
    mse_relIncr = []
    r2_relIncr_both = []
    mse_relIncr_both = []
    n_aggregs = []
    n_aggregs_feat = []

    for i in range(args.n_reps):
        # generate_features_from_droughts
        logging.warning(f'Iteration: {i}')
        logging.warning(f"Current time: {datetime.datetime.now()}")
        df_train,df_test,y_df_train,y_df_test = generate_correlated_features(n_feat=args.n_feats, train_dim=args.train_dim, test_dim=250, seed=i, n_tasks=args.n_tasks, noise=args.noise)
        logging.debug(df_train.shape,df_test.shape,y_df_train.shape,y_df_test.shape)
        logging.debug(df_train,df_test,y_df_train,y_df_test)

        clustering = NonLinCTA(df=df_train, targets_df=y_df_train, eps=args.eps1, n_val=-1, neigh=0) # n_val=-1/5
        output = clustering.compute_target_clusters()
        logging.warning(output)
        n_aggregs.append(len(output))

        for aggreg in output:
            for tas in aggreg:
                logging.debug(tas)

                regr_aggr = LinearRegression().fit(df_train, y_df_train.iloc[:,aggreg].mean(axis=1))
                regr_single = LinearRegression().fit(df_train, y_df_train.iloc[:,tas])

                actual_r2_single = regr_single.score(df_test, y_df_test.iloc[:,tas])
                actual_r2_aggr = regr_aggr.score(df_test, y_df_test.iloc[:,tas])
                actual_mse_single = mean_squared_error(y_df_test.iloc[:,tas],regr_single.predict(df_test))
                actual_mse_aggr = mean_squared_error(y_df_test.iloc[:,tas],regr_aggr.predict(df_test))
                
                r2_single.append(actual_r2_single)
                r2_aggr.append(actual_r2_aggr)
                r2_relIncr.append(100*(actual_r2_aggr-actual_r2_single)/actual_r2_single)
                mse_single.append(actual_mse_single)
                mse_aggr.append(actual_mse_aggr)
                mse_relIncr.append(-100*(actual_mse_aggr-actual_mse_single)/actual_mse_single)

                ### NonLinCFA_new on the aggregated target
                feature_clustering = NonLinCFA_new(df=df_train, target=y_df_train.iloc[:,aggreg].mean(axis=1), eps=args.eps2, n_val=5, neigh=0, verbose=False) # n_val=-1/5
                feature_output = feature_clustering.compute_feature_clusters()

                red_train = pd.DataFrame()
                red_test = pd.DataFrame()
                i=0
                for out in feature_output:
                    red_train[str(i)] = df_train.loc[:,out].mean(axis=1)
                    red_test[str(i)] = df_test.loc[:,out].mean(axis=1)
                    i += 1
                regr_aggr_both = LinearRegression().fit(red_train, y_df_train.iloc[:,aggreg].mean(axis=1))
                actual_r2_aggr_both = regr_aggr_both.score(red_test, y_df_test.iloc[:,tas])
                actual_mse_aggr_both = mean_squared_error(y_df_test.iloc[:,tas],regr_aggr_both.predict(red_test))

                r2_aggr_both.append(actual_r2_aggr_both)
                mse_aggr_both.append(actual_mse_aggr_both)
                r2_relIncr_both.append(100*(actual_r2_aggr_both-actual_r2_single)/actual_r2_single)
                mse_relIncr_both.append(-100*(actual_mse_aggr_both-actual_mse_single)/actual_mse_single)
                n_aggregs_feat.append(len(feature_output))

    logging.warning(f"Current time: {datetime.datetime.now()}")
    logging.warning(f'R2:\n\tsingle: {np.mean(r2_single)}, aggregate: {np.mean(r2_aggr)}, CI: {print_95CI(r2_relIncr)}%')
    logging.warning(f'MSE:\n\tsingle: {np.mean(mse_single)}, aggregate: {np.mean(mse_aggr)}, CI: {print_95CI(mse_relIncr)}%')
    logging.warning(f'Aggregations: {print_95CI(n_aggregs)}')

    logging.warning(f'FEATURES--> R2:\n\tsingle: {np.mean(r2_single)}, aggregate: {np.mean(r2_aggr_both)}, CI: {print_95CI(r2_relIncr_both)}%')
    logging.warning(f'MSE:\n\tsingle: {np.mean(mse_single)}, aggregate: {np.mean(mse_aggr_both)}, CI: {print_95CI(mse_relIncr_both)}%')
    logging.warning(f'Aggregations: {print_95CI(n_aggregs_feat)}')
    #print(np.mean(r2_relIncr))
    #print(np.std(r2_relIncr))