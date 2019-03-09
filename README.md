This project involves the analysis of subset of the UCI Dataset for Social Media Buzz: - 
[ Prédictions d’activité dans les réseaux sociaux en ligne (F. Kawala, A. Douzal-Chouakria, E. Gaussier, E. Dimert), In Actes de la Conférence sur les Modèles et l′Analyse des Réseaux : Approches Mathématiques et Informatique (MARAMI), pp. 16, 2013.]
https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+ 

The Dataset has two parts - one which has been used for Classification and the other which has been used for regression. 
The goal of this project is to mainly find out the optimum machine learning model or a ensemble of models 
that can accurately classify a social media data-point to be a buzz or not. The dataset has 77 features and over 1/2 million data points.

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

We begin by analysing the dataset to see any missing values and to check whether all the features are numerical in order to run our machine learning models. When the data seems fine, we select GridSearchCV for evaluating the correct parameter for Linear Regression, Polynomial Regression, Ridge Regression, Lasso Regression, Linear Support Vector Regression, Support Vector Regression with Kernel trick and finally KNN Nearest Neighbor Regression. After our analysis of training and test scores of various models and also the cross validation scores, we decided that Lasso Regression was the best model to go with as its generalization was bit better than other models having the same training and test scores.
We ran Lasso Regression on whole dataset, using GridSearchCV again to confirm the best parameter and found a decent fit. We examined the co-efficient values and many of them have been reduced to zero as per the Lasso properties. Overall, looking at the plot, this seems a proper fit.

Further Analysis: Classification

Once we have created individual models and seen the scores, not we will test ensembling, principal component analysis and deep learning.

Ensemble models :

It is observed that Soft voting gives overall higher accuracy scores. LinearSVC with Hard voting has the highest accuracy score within the Hard voting classifier of 96.33% and Logistic Regression with Soft voting has the highest accuracy score within the Soft voting classifier of 96.13%. Implemented KNeighborsClassifier and LogisticRegression with Bagging, LinearSVC and DTree Classifier with pasting & DTree Classifier LogisticRegression using Adaboost. GradientBoostingClassifier with max_depth=3 had good accuracy. The best Ensemble model is Decision Tree classifier with Adaboost with 100% Train accuracy and 96.36% test accuracy.

PCA :

Standard scaler is recommended for PCA hence used that for scale the original X data and then applied PCA on it. Ran then PCA model with 0.95 variance and got 15 reduced components. Ran all the models from Projects 2 on the reduced data after PCA. After comparing these results with those of previous project, we observed that the accuracy of the train accuracy improved slightly after performing KNN on the PCA reduced dataset. Also observed approx 1% increase in the accuracy after running Kernel SVM on the PCA reduced dataset. However, the test accuracy has reduced for other models like Logistic regression, Decision tree and Kernalized SVM after using PCA reduced dataset. The reduction in accuracy may be attributed to the fact that the underlying sampling data might have changed from Project 2 to Project 3 that is the 10% sample of Project 2 might be slightly different from that selected in Project 3. Best model after PCA is Kernal SVM.

Deep learning:

Implemented the The deep learning model having Objective Function/Loss Function of Binary cross-entropy Ran grid search on KerasClassifier and found the best parameter is batch_size of 20 and epochs of 50. Built a model on this and got the accuracy of 96.84%

Further Analysis: Regression

For ensemble modelling, like all the individual regression models, we have done MinMaxScaler and before PCA we have have used StandardScaler - the reason is we wanted to transform the data to zero mean and unit variance. After PCA decomposition, 77 features are reduced to 18 principal components.
After the reduced data is fed into the models after PCA decomposition, from the results we can see that for Linear, Ridge and Lasso Regression the training and testing score are almost the same and all three are a fair fit. Now the reduction in accuracy may be attributed to the fact that the underlying sampling data might have changed from Project 2 to Project 3 - i.e. the 10% sample of Project 2 is slightly different from Project 3.
The best model after application of PCA seems to be KNNRegressor which has the highest training and test accuracy. Defnitely after the application of PCA, the results of KNNRegressor are slightly better as Training score and Testing score are more closer.
Among the Ensemble models, Adaboosting Decision Tree with maximum depth = 12 seems to be the best but it seems to be slightly overfitted due to the gap between training and testing score.
The next best ensemble model is LinearSVR pasting due to closeness of Training and Test score.
We tried to implement two Deep Learning models by arranging different node layers, but both of them gave a very high mean-square error, hence we have discarded the deep learning models from our consideration.
