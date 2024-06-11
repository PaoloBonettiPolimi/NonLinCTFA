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
import scipy.io

def load_school(d=27):#d=139 
    df_list = []
    y_df_list = []
    dataset = scipy.io.loadmat('/Users/paolo/Documents/MultiLinCFA/MultiTaskLearning/data/school.mat')
    FEATURES_COLUMNS = ['Year_1985','Year_1986','Year_1987','FSM','VR1Percentage','Gender_Male','Gender_Female','VR_1','VR_2','VR_3',
                    'Ethnic_ESWI','Ethnic_African','Ethnic_Arabe','Ethnic_Bangladeshi','Ethnic_Carribean','Ethnic_Greek','Ethnic_Indian',
                    'Ethnic_Pakistani','Ethnic_Asian','Ethnic_Turkish','Ethnic_Others','SchoolGender_Mixed','SchoolGender_Male',
                    'SchoolGender_Female','SchoolDenomination_Maintained','SchoolDenomination_Church','SchoolDenomination_Catholic',
                    'Bias']
    X_df=pd.DataFrame(dataset['X'][:,0][0],columns=FEATURES_COLUMNS)#.iloc[:20,:]
    y_df=pd.DataFrame(dataset['Y'][:,0][0],columns=['task_0'])#.iloc[:20,:]
    #X_df['School'] = 1
    #y_df['School'] = 1
    df_list.append(X_df)
    y_df_list.append(y_df)
    print(d)
    for i in range(1,d):
        X_df_i=pd.DataFrame(dataset['X'][:,i][0],columns=FEATURES_COLUMNS)#.iloc[:20,:]
        df_list.append(X_df_i)
        #X_df = X_df.append(X_df_i,ignore_index=True)

        y_df_i=pd.DataFrame(dataset['Y'][:,i][0], columns=['task_'+str(i)])#.iloc[:20,:]  ['Exam_Score_'+str(i)]
        y_df_list.append(y_df_i)
        #y_df_i['School'] = i+1  
        #y_df = y_df.append(y_df_i,ignore_index=True)
    return df_list,y_df_list

def create_list_for_clustering(df_list,y_df_list):
    df_list_forClustering = []
    
    maximum = 0
    for dtf in df_list:
        maximum = max(maximum,dtf.shape[0])

    for i in range(df_list[0].shape[1]):
        df_full = pd.DataFrame({"col1":["value"]*maximum})
        for j in range(len(df_list)):
            col = df_list[j].iloc[:,i].reset_index(drop=True)
            df_full['task_'+str(j)] = col
        df_list_forClustering.append(df_full.iloc[:,1:])
        
    y_df_list_forClustering = pd.DataFrame({"col1":["value"]*maximum})

    for dd in y_df_list:
        y_df_list_forClustering = pd.concat((y_df_list_forClustering,dd),axis=1)
    y_df_list_forClustering = y_df_list_forClustering.iloc[:,1:]
  
    return df_list_forClustering,y_df_list_forClustering

### test on original tasks

def aggr_model_test_onSingle(df_list, y_df_list, df_list_clustering, y_df_list_clustering, res, task_model, number_of_splits=5, test_size=0.3):
    
    scores = []
    ss = ShuffleSplit(n_splits=number_of_splits, test_size=test_size, random_state=42)
    
    # for each task look for its aggregated model and test on itself
    for i in range(len(df_list)):
        
        y_df = y_df_list[i] ### has its 123 values
        x_df = df_list[i] ### has its 123 values
        
        name = 'task_'+str(i)
        for elem in res:
            if name in res[elem]:
                actual_aggr_list = res[elem]
                #print(actual_aggr_list)
        y_df_aggr = pd.DataFrame(y_df_list_clustering.loc[:,actual_aggr_list].mean(axis=1),columns=['aggr_target']).dropna() ### has 175 values
        
        x_df_aggr = pd.DataFrame()
        kk = 0
        for df_clust in df_list_clustering:
            x_df_aggr[df_list[0].columns[kk]] = df_clust.loc[:,actual_aggr_list].mean(axis=1)
            kk += 1 
        x_df_aggr = x_df_aggr.dropna()
        
        #print(x_df_aggr,y_df_aggr)
        
        # for each of the five repetitions
        for train_index, test_index in ss.split(x_df):
        
            X_train, X_test = x_df.iloc[train_index,:], x_df.iloc[test_index,:]
            y_train, y_test = y_df.iloc[train_index,:], y_df.iloc[test_index,:]
            X_train_aggr, X_test_aggr = x_df_aggr.iloc[train_index,:], x_df_aggr.iloc[test_index,:]
            y_train_aggr, y_test_aggr = y_df_aggr.iloc[train_index,:], y_df_aggr.iloc[test_index,:]

            model = task_model
            model.fit(x_df_aggr,y_df_aggr) #model.fit(X_train_aggr,y_train_aggr)
            y_pred = model.predict(X_test_aggr)
            
            score = np.sqrt(mean_squared_error(y_test, y_pred))/69 #(max(y_train.values)-min(y_train.values))
            # for linear regression conditioning
            if score<1:
                scores.append(score)#(score[0])
    return scores

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

    df_list,y_df_list = load_school(d=27)#139, 27
    df_list_clustering, y_df_list_clustering = create_list_for_clustering(df_list,y_df_list)

    ### MultiLinCFA --- eps=0 NonLinCTA_link
    my_class = NonLinCTAext(df_list_clustering,y_df_list_clustering)
    #my_class = NonLinCTA_link(df_list_clustering,y_df_list_clustering)
    my_class.compute_clusters()
    res = my_class.get_clusters()
    
    scores_LR = aggr_model_test_onSingle(df_list, y_df_list, df_list_clustering, y_df_list_clustering, res, LinearRegression(), number_of_splits=5, test_size=0.3)

    mod = SVR(kernel='rbf', C=1e1, gamma=0.1,epsilon=0.01)
    scores_SVM = aggr_model_test_onSingle(df_list, y_df_list, df_list_clustering, y_df_list_clustering, res, mod, number_of_splits=5, test_size=0.3)


    
        