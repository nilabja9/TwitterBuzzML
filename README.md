This project involves the analysis of subset of the UCI Dataset for Social Media Buzz: - 
[ Prédictions d’activité dans les réseaux sociaux en ligne (F. Kawala, A. Douzal-Chouakria, E. Gaussier, E. Dimert), In Actes de la Conférence sur les Modèles et l′Analyse des Réseaux : Approches Mathématiques et Informatique (MARAMI), pp. 16, 2013.]
https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+ 

The Dataset has two parts - one which has been used for Classification and the other which has been used for regression. 
The goal of this project is to mainly find out the optimum machine learning model or a ensemble of models 
that can accurately classify a social media datapoint to be a buzz or not. The dataset has 77 features and over 1/2 million datapoints.

Classification Summary and Findings:

Preprocessing: 

• Checking missing values – no missing values found 
• Renamed the column names based on the feature labels 
• Cleaned the buzz column to convert it to binary 
• Generated X and y – kept all the features in X dataset and kept the buzz column in y 

Classification Strategy: 

As the dataset is imbalanced, we selected AUC has the appropriate evaluation strategy
Best parameters for each model: KNN - n_neighbors: 20 Logistic Regression - C: 100 Linear SVC - C: 10 SVM RBF kernel - C': 100, 'gamma': 0.1 Decision Tree - Max_depth – 3
Best Model: Based on Train AUC score and Test AUC score – Linear SVC is the best model
Results after running Linear SVC on the entire dataset: Model name - LinearSVC Model parameter - C = 1 Train accuracy - 0.9913051647476957 Test accuracy - 0.992104740282261 Train auc score - 0.9559502123242883 Test auc score - 0.941574081838001
Buzz predicted percentage = 19.46%

Regression Strategy:

We begin by analyzing the dataset to see any missing values and to check whether all the features are numerical in order to run our machine learning models. When the data seems fine, we select GridSearchCV for evaluating the correct parameter for Linear Regression, Polynomial Regression, Ridge Regression, Lasso Regression, Linear Support Vector Regression, Support Vector Regression with Kernel trick and finally KNN Nearest Neighbor Regression. After our analysis of training and test scores of various models and also the cross validation scores, we decided that Lasso Regression was the best model to go with as its generalization was bit better than other models having the same training and test scores.
We ran Lasso Regression on whole dataset, using GridSearchCV again to confirm the best parameter and found a decent fit. We examined the co-efficient values and many of them have been reduced to zero as per the Lasso properties. Overall, looking at the plot, this seems a proper fit.
