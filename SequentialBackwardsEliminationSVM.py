from math import gamma
from comet_ml import Experiment

from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.sparse.construct import rand
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

#read dataset from csv-file
data = pd.read_csv('')
data.drop(columns=['Unnamed: 0'], inplace=True)
y = data['label']
X = data.drop(columns=['label'], axis=1)

#apply normalization column-wise
columnnames = X.columns.to_numpy()
Xnorm = pd.DataFrame(normalize(X, norm="max",axis = 0, copy=False),columns=columnnames)

#declare test-split size and number of folds for cross validation
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

#declare context cue features varying in feature selection
nonstaticfeatures = ['retweet_count','user_verified','user_friends_count','user_followers_count','user_favourites_count','tweet_source','geo_coordinates_available','num_hashtags','num_mentions','num_urls','num_media']
staticfeatures = X.columns.tolist()

#find names of columns containing non context cue features
staticfeatures = ([i for i in staticfeatures if i not in nonstaticfeatures])

#split dataset into test- and train-splits
cv_gen = sss.split(Xnorm,y)
cv = list(cv_gen)

#perform Sequential Feature Selection with number on included context cue features from 1 to 10, as there are 10 context cue features
for i in range (0,10):
    #setup experiment on comet.ml 
    exp= Experiment(    api_key="",
        project_name="",    workspace="")
    
    #instantiate Support Vector Machine classifier, using Radial Basis Function as kernel
    svc = SVC(C=10,gamma=0.0001,kernel="rbf",random_state=42) 

    k = 5711+i

    #set experiment name on comet.ml
    exp.set_name('sequentialForwardk'+str(k))

    #instantiate feature selector using the Support Vector Machine classifier with 5-fold stratified cross validation. Scoring can be set to accuracy or F1 to evaluate feature performance on accuracy and F1-Score respectively
    sfs = SequentialFeatureSelector(svc, k_features=k, forward=False, scoring='accuracy', cv=cv,fixed_features=staticfeatures, verbose=2, n_jobs=-1)

    #perform Sequential Feature Selection on dataset
    sfs.fit(X,y)

    #record the names of selected context cue features
    selectedfeatures = ([i for i in sfs.k_feature_names_ if i not in staticfeatures])
    results = {
        'selectedFeatures':str(selectedfeatures),
        'score':str(sfs.k_score_)
    }

    #log the results of the experiment on comet.ml
    exp.log_metrics(results)

    #export the metrics of the feature selector
    metric_dict = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    metric_dict.to_csv('', index=False)


