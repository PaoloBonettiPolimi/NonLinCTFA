import logging
logging.basicConfig(filename='../MultiLinCFA/synthExp_varySamples3.log',
                    filemode='a',
                    level=logging.WARN)
import pandas as pd
import sys
import numpy as np
sys.path.append("../MultiLinCFA")
from MultiLinCFA import NonLinCTA,NonLinCFA_new
from MultiLinCFA_extended import NonLinCTAext
from MultiLinCFA_linkage import NonLinCTA_link
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import random
sys.path.append("../MultiTaskLearning")
from mult_ind_SVM import mult_ind_SVM
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

#def preprocessing(X,Y):
#    """
#    Prepare the dataset for the MTL algorithms
#    """
#    X_process=np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
#    y_process=np.concatenate((Y[:,0].reshape(Y.shape[0],1),np.ones((Y.shape[0],1))),axis=1)
#    for l in range(2,Y.shape[1]+1):
#        X_l=np.concatenate((X,np.ones((X.shape[0],1))*l),axis=1)               
#        X_process=np.append(X_process,X_l,axis=0)
#        y_l = np.concatenate((Y[:,0].reshape(Y.shape[0],1),l*np.ones((Y.shape[0],1))),axis=1)
#        y_process=np.append(y_process,y_l,axis=0)
#    return X_process, y_process
import scipy
def load_sarcos_dataset(set_size=1000):
    """
    Load SARCOS dataset and select the first 2000 samples for computing reasons
    """
    # Load training set
    sarcos_train = scipy.io.loadmat('/Users/paolo/Documents/MultiLinCFA/MultiTaskLearning/data/sarcos_inv.mat')
    # Inputs  (7 joint positions, 7 joint velocities, 7 joint accelerations)
    Xtrain = sarcos_train["sarcos_inv"][:, :21]
    # Outputs (7 joint torques)
    Ytrain = sarcos_train["sarcos_inv"][:, 21:]

    # Load test set
    sarcos_test = scipy.io.loadmat('/Users/paolo/Documents/MultiLinCFA/MultiTaskLearning/data/sarcos_inv_test.mat')
    Xtest = sarcos_test["sarcos_inv_test"][:, :21]
    Ytest = sarcos_test["sarcos_inv_test"][:, 21:]

    X = np.concatenate((Xtrain,Xtest),axis=0)
    Y = np.concatenate((Ytrain,Ytest),axis=0)

    return X[:set_size,:],Y[:set_size,:]

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
    logging.warn(f"noise: {args.noise}, n_reps: {args.n_reps}, n_tasks: {args.n_tasks}, n_feats: {args.n_feats}, train_dim: {args.train_dim}, eps: {args.eps1,args.eps2}")
    ##################### experiments ########################

    ### basic configuration
    X,Y = load_sarcos_dataset(set_size=1000)
    print(X.shape,Y.shape)
    X_df = pd.DataFrame(X)
    X_df = (X_df-X_df.mean(axis=0))/X_df.std(axis=0)
    Y_df = pd.DataFrame(Y)
    Y_df = (Y_df-Y_df.mean(axis=0))/Y_df.std(axis=0)

    n_reps = 5
#output = [[0], [1], [2], [3,6], [4], [5]]

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
    nrmse_single = []
    nrmse_aggr = []
    nrmse_aggr_both = []

    for i in range(n_reps):
        
        np.random.seed(i)
        msk = np.random.rand(len(X_df)) < 0.7
        df_train = X_df.loc[:699,:] #X_df[msk].reset_index(drop=True)
        df_test = X_df.loc[700:,:]#X_df[~msk].reset_index(drop=True)
        y_df_train = Y_df.loc[:699,:]#Y_df[msk].reset_index(drop=True)
        y_df_test = Y_df.loc[700:,:]#Y_df[~msk].reset_index(drop=True)
        
        clustering = NonLinCTA(df=df_train, targets_df=y_df_train, eps=-0.01, n_val=-1, neigh=0, verbose=True) # n_val=-1/5
        output = clustering.compute_target_clusters()
        print(output)
        print(len(output))
        n_aggregs.append(len(output))
        
        for aggreg in output:
            for tas in aggreg:
                print(tas)
        
                regr_aggr = LinearRegression().fit(df_train, y_df_train.iloc[:,aggreg].mean(axis=1)) #SVR(kernel='rbf', C=C, gamma=0.1,epsilon=0.01)
                regr_single = LinearRegression().fit(df_train, y_df_train.iloc[:,tas])
        
                actual_r2_single = regr_single.score(df_test, y_df_test.iloc[:,tas])
                actual_r2_aggr = regr_aggr.score(df_test, y_df_test.iloc[:,tas])
                actual_mse_single = mean_squared_error(y_df_test.iloc[:,tas],regr_single.predict(df_test))
                actual_mse_aggr = mean_squared_error(y_df_test.iloc[:,tas],regr_aggr.predict(df_test))
                normaliz = (max(y_df_train.values.reshape(-1,1))-min(y_df_train.values.reshape(-1,1)))
                actual_nrmse_single = np.sqrt(actual_mse_single)/normaliz
                actual_nrmse_aggr = np.sqrt(actual_mse_aggr)/normaliz

                r2_single.append(actual_r2_single)
                r2_aggr.append(actual_r2_aggr)
                r2_relIncr.append(100*(actual_r2_aggr-actual_r2_single)/actual_r2_single)
                mse_single.append(actual_mse_single)
                mse_aggr.append(actual_mse_aggr)
                mse_relIncr.append(-100*(actual_mse_aggr-actual_mse_single)/actual_mse_single)
                nrmse_single.append(actual_nrmse_single)
                nrmse_aggr.append(actual_nrmse_aggr)
        
                ### NonLinCFA_new on the aggregated target
                print(df_train.shape,y_df_train.iloc[:,aggreg].mean(axis=1).shape)
                feature_clustering = NonLinCFA_new(df=df_train, target=y_df_train.iloc[:,aggreg].mean(axis=1), eps=0, n_val=5, neigh=0, verbose=False) # n_val=-1/5
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
                actual_nrmse_aggr_both = np.sqrt(actual_mse_aggr_both)/normaliz
        
                r2_aggr_both.append(actual_r2_aggr_both)
                mse_aggr_both.append(actual_mse_aggr_both)
                r2_relIncr_both.append(100*(actual_r2_aggr_both-actual_r2_single)/actual_r2_single)
                mse_relIncr_both.append(-100*(actual_mse_aggr_both-actual_mse_single)/actual_mse_single)
                n_aggregs_feat.append(len(feature_output))
                nrmse_aggr_both.append(actual_nrmse_aggr_both)

        