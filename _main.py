import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from datetime import datetime
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.lda import LDA
from sklearn.mixture import GaussianMixture
import numpy as np
import seaborn as sns
from matplotlib.pyplot import savefig
# Config file. We will be using directions from 
# https://medium.com/autonomous-agents/how-to-train-your-neuralnetwork-for-wine-tasting-1b49e0adff3a
# https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn#step-5

data_dir = "C:\\Users\\haparashar\\Documents\\WORK\\EXTRAS\\Wine data - Machine Learning Guild\\"
train_file_name = "train_data.csv"
test_file_name = "test_data_withClass.csv"
wine_file_name = "wine.csv"

classifiers = [GaussianNB(), 
               DecisionTreeClassifier(), 
               LogisticRegression(), 
               SVC(), 
               RandomForestClassifier(), 
               GradientBoostingClassifier(),
               GaussianMixture(),
               QuadraticDiscriminantAnalysis(), 
               LDA()]

hyperparam = {"DecisionTreeClassifier" : {'criterion': ['gini', 'entropy']},
             "SVC" : {'kernel':('linear', 'rbf'), 
                      'C' :[0.001, 0.01, 0.1, 1, 2, 5, 10],
                     'gamma' : [0.001, 0.01, 0.05, 0.10, 0.5]},
            "RandomForestClassifier" : {'max_features' : ['auto', 'sqrt', 'log2'],
                                       'max_depth': [3, 2]},
            "GradientBoostingClassifier" : {'learning_rate' : [0.01, 0.05, 0.1],
                                            'n_estimators' : [10, 20],
                                            'max_depth' : [3]},
            "GaussianMixture" : {"n_components" : [2],
                                "covariance_type" : ["full", "tied", "diag", "spherical"],
                                 "n_init" : [2, 3],
                                 "init_params" : ["kmeans", "random"],
                                 "max_iter" : [200],
                                 "tol" : [1e-4],
                                 "random_state" : [10, 20 ,30]},
            "QuadraticDiscrimminantAnalysis" : {"n_components" : [2]}}

output = open("output.txt", "a")
output.write(("=")*20 + "\n" + str(datetime.now().now()) + "\n")
#==============================================================================
# READING IN THE INPUT FILES
#==============================================================================
# Creating custom scoring function
def confusion_scorer(ground_truth, predictions) :
    mx = confusion_matrix(ground_truth, predictions)
    return ((mx[0,0] + mx[1,1])/mx.sum())

score_func = make_scorer(confusion_scorer, greater_is_better = True)

# Readin the files
train = pd.read_csv(data_dir + train_file_name)
test = pd.read_csv(data_dir + test_file_name)
wine = pd.read_csv(data_dir + wine_file_name, header = None, names = train.columns)

#Getting what we need to give as output
Y_test = test.wine_class 
X_test = test.drop('wine_class', axis = 1)
X_train = train.drop('wine_class', axis = 1)
Y_train = train.wine_class

#Analysing the changes made by the HRT Guys
wine.rename(columns = {"wine_class" : "wine_class_uci"}, inplace = True)
wine = pd.merge(wine, test, how = "left", on = list(train.columns[1:]))
wine = pd.merge(wine, train, how = "left", on = list(train.columns[1:]))
wine.loc[np.isnan(wine.wine_class_x), "wine_class_x"] = wine.loc[np.isnan(wine.wine_class_x), "wine_class_y"]
wine = wine.drop("wine_class_y", axis =1) 
wine['changed'] = abs(wine.wine_class_uci - wine.wine_class_x)

sns.pairplot(wine[wine["wine_class_uci"]==1],hue="changed").savefig("Analysis1.png")
 # Getting the files together

#==============================================================================
# PREPROCESSING THE FILES
#==============================================================================

# Now, we won't use simply scale to transform, because The reason is that we won't be able 
# to perform the exact same transformation on the test set.
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

for clf in classifiers :
    # Creating a pipeline for transformation
    model = clf.__class__.__name__
    if str(clf.__class__.__name__) in hyperparam :
        clf = GridSearchCV(clf, param_grid = hyperparam[str(clf.__class__.__name__)], 
                                                        scoring = score_func, cv = 10)

    clf.fit(X_train, Y_train)

    pred_train, pred_test = clf.predict(X_train), clf.predict(X_test)
    accuracy_train, accuracy_test = confusion_scorer(Y_train, pred_train), confusion_scorer(Y_test, pred_test)
    
    output.write(str(model) + " : Train = " + str(accuracy_train) + "   Test = " + str(accuracy_test)  +"\n")



output.close()