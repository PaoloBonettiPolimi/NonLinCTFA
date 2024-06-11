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

### print 95% CI considering the distribution to be gaussian
def print_95CI(mylist):
    return str(round(np.mean(mylist),6))+'±'+str(round(1.96*np.std(mylist)/np.sqrt(len(mylist)),6))

### compute 95% CI considering the distribution to be gaussian
def compute_95CI(mylist):
    #return str(round(np.mean(mylist),6))+'±'+str(round(1.96*np.std(mylist)/np.sqrt(3000),6))
    return np.mean(mylist)-1.96*np.std(mylist)/np.sqrt(3000),np.mean(mylist)+1.96*np.std(mylist)/np.sqrt(len(mylist))

def load_school_dataset():
    """
    Load School dataset and select the first 27 tasks  for computing reasons
    """
    dataset = scipy.io.loadmat('data/school.mat')
    FEATURES_COLUMNS = ['Year_1985','Year_1986','Year_1987','FSM','VR1Percentage','Gender_Male','Gender_Female','VR_1','VR_2','VR_3',
                'Ethnic_ESWI','Ethnic_African','Ethnic_Arabe','Ethnic_Bangladeshi','Ethnic_Carribean','Ethnic_Greek','Ethnic_Indian',
                'Ethnic_Pakistani','Ethnic_Asian','Ethnic_Turkish','Ethnic_Others','SchoolGender_Mixed','SchoolGender_Male',
                'SchoolGender_Female','SchoolDenomination_Maintained','SchoolDenomination_Church','SchoolDenomination_Catholic',
                'Bias']
    
    # Dataframe representation
    X_df=pd.DataFrame(dataset['X'][:,0][0],columns=FEATURES_COLUMNS)
    y_df=pd.DataFrame(dataset['Y'][:,0][0],columns=['Exam_Score'])
    X_df['School'] = 1
    y_df['School'] = 1
        
    #d = X_df.shape[1]-1
    d = 139
    print(d)
    for i in range(1,d):
        X_df_i=pd.DataFrame(dataset['X'][:,i][0],columns=FEATURES_COLUMNS)
        X_df_i['School'] = i+1  
        X_df = X_df.append(X_df_i,ignore_index=True)

        y_df_i=pd.DataFrame(dataset['Y'][:,i][0],columns=['Exam_Score'])
        y_df_i['School'] = i+1  
        y_df = y_df.append(y_df_i,ignore_index=True)  
        
    return X_df, y_df

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
    parser.add_argument("--dataset", default='school', type=str)

    args = parser.parse_args()
    print(args)
    
    logging.warning(f"Current time: {datetime.datetime.now()}")

    ##################### experiments ########################

    if args.dataset=='school':


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