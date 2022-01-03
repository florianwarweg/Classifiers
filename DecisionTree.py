from matplotlib.colors import from_levels_and_colors
import numpy as np
from numpy.random.mtrand import random_sample 
import pandas as pd 
import graphviz
import matplotlib.pyplot as plt 
from matplotlib.pyplot import savefig, text, tight_layout
from scipy.sparse.construct import rand
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#read dataset from csv-file
df = pd.read_csv('')

#drop columns from dataframe non considered in context cue feature selection
df.drop(columns=['Unnamed: 0','fake_news_category_2','text','tweet_source'], inplace=True)

X = df.drop(columns=['label'])
y = df['label']
columnnames = X.columns.tolist()

#split dataset into train- and test-splits with test-split containing 30% of samples in the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#instantiate Decision Tree Classifier, using Gini impurity as a function for measuring quality of splits
clf_dt = DecisionTreeClassifier(criterion='gini',random_state=42)
#train Decision Tree Classifier on train-split of dataset
clf_dt.fit(X_train, y_train)

#apply Minimal Cost-Complexity Pruning and save alpha values of paths
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

alpha_loop_values = []

#record performance for all alpha values saved during Minimal Cost-Complexity Pruning using 5-fold cross validation
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(criterion='gini',random_state=42, ccp_alpha=ccp_alpha)
    scores = scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])


alpha_results = pd.DataFrame(alpha_loop_values, 
    columns=['alpha', 'mean_accuracy', 'std'])


#select best alpha value in terms of accuracy
ideal_alpha = alpha_results.at[alpha_results['mean_accuracy'].idxmax(),'alpha']

#instantiate Decision Tree Classifier, using Gini impurity as a function for measuring quality of splits and the ideal alpha found during cross validation
clf_dt_pruned = DecisionTreeClassifier(criterion='gini', random_state=42, ccp_alpha=ideal_alpha)

accuracy = []
f1 = []
recall = []
precision = []

#split dataset five times into train- and test-splits with test-split containing 30% of samples in the dataset. The precentages of classed in the dataset is preserved.
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

X=X.to_numpy()
y=y.to_numpy()

sss.get_n_splits(X, y)

#trains and evaluates the performance of the Decision Tree classifier using ideal alpha value using 5-fold cross validation
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_pred = clf_dt_pruned.fit(X_train,y_train).predict(X_test)
    
    precision.append(metrics.precision_score(y_test,y_pred))
    recall.append(metrics.recall_score(y_test,y_pred))
    f1.append(metrics.f1_score(y_test,y_pred))
    accuracy.append(clf_dt_pruned.score(X_test, y_test))

   
results = {
        "accuracy":sum(accuracy)/len(accuracy),
        "f1":sum(f1)/len(f1),
        "recall":sum(recall)/len(recall),
        "precision":sum(precision)/len(precision),
        }        

#export tree as text represenation
textTree = tree.export_text(clf_dt_pruned, feature_names=columnnames)

text_file = open('', "w")
n = text_file.write(textTree)
text_file.close()

#export tree as DOT file
dot_data = tree.export_graphviz(clf_dt_pruned, 
    out_file='',
    feature_names=columnnames,
    class_names=["True_News","Fake_News"],
    impurity=True,
    filled=True,
    node_ids=True)



  
