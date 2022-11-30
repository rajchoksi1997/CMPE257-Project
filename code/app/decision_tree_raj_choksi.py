
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pickle


def decisionTreeWtihSmoteUpsampledData(x_train_ures_SMOTE,y_train_ures_SMOTE,x_test,y_test):
    dt_Classifier_up_smote = tree.DecisionTreeClassifier(criterion='gini', max_depth=70)
    dt_Classifier_up_smote.fit(x_train_ures_SMOTE,y_train_ures_SMOTE)
    #training prediction
    y_train_pred = dt_Classifier_up_smote.predict(x_train_ures_SMOTE)
    #testing prediction
    y_prediction_up_smote = dt_Classifier_up_smote.predict(x_test)
    print('training f1 score:',f1_score(y_train_ures_SMOTE, y_train_pred, average='weighted'))
    print('Analysis of Decision Tree Algorithm with SMOTE upsampled data:\n')
    print(confusion_matrix(y_test, y_prediction_up_smote))
    print(classification_report(y_test, y_prediction_up_smote))
    print('Accuracy:',accuracy_score(y_test, y_prediction_up_smote))
    print('Precision:',precision_score(y_test, y_prediction_up_smote, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction_up_smote, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction_up_smote, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction_up_smote))
    pickle.dump(dt_Classifier_up_smote, open('dt_Classifier_up_smote.pkl', 'wb'))

def decisionTreeWithPCAandUpsampledData(X_train_upsampled_pca,y_train_ures_SMOTE,X_test_upsampled_transformed_pca,y_test):
    dt_classifier_smote_upsampling_pca = tree.DecisionTreeClassifier(criterion='gini')
    dt_classifier_smote_upsampling_pca.fit(X_train_upsampled_pca,y_train_ures_SMOTE)
    #training prediction
    y_train_pred = dt_classifier_smote_upsampling_pca.predict(X_train_upsampled_pca)
    #testing prediction
    y_prediction_smote_upsampling_pca = dt_classifier_smote_upsampling_pca.predict(X_test_upsampled_transformed_pca)
    print('Training F1 Score:',f1_score(y_train_ures_SMOTE, y_train_pred, average='weighted'))
    print('Analysis of Decision Tree Algorithm with PCA and SMOTE Upsampled data:\n')
    print(confusion_matrix(y_test, y_prediction_smote_upsampling_pca))
    print(confusion_matrix(y_test, y_prediction_smote_upsampling_pca))
    print(classification_report(y_test, y_prediction_smote_upsampling_pca))
    print('Accuracy:',accuracy_score(y_test, y_prediction_smote_upsampling_pca))
    print('Precision:',precision_score(y_test, y_prediction_smote_upsampling_pca, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction_smote_upsampling_pca, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction_smote_upsampling_pca, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction_smote_upsampling_pca))
    pickle.dump(dt_classifier_smote_upsampling_pca, open('dt_classifier_smote_upsampling_pca.pkl', 'wb'))

def decisionTreeWithLDAandUpsampledData(X_train_upsampled_LDA,y_train_ures_SMOTE,X_test_upsampled_LDA_transformed,y_test):
    dt_classifier_smote_upsampled_lda = tree.DecisionTreeClassifier(criterion='gini')
    dt_classifier_smote_upsampled_lda.fit(X_train_upsampled_LDA,y_train_ures_SMOTE)
    #training prediction
    y_train_pred = dt_classifier_smote_upsampled_lda.predict(X_train_upsampled_LDA)
    #testing prediction
    y_pred_smote_upsampled_lda = dt_classifier_smote_upsampled_lda.predict(X_test_upsampled_LDA_transformed)
    print('F1 Score:',f1_score(y_train_ures_SMOTE, y_train_pred, average='weighted'))
    print('Analysis of Decision Tree Algorithm with  LDA and SMOTE Upsampled Data:\n')
    print(confusion_matrix(y_test, y_pred_smote_upsampled_lda))
    print(confusion_matrix(y_test, y_pred_smote_upsampled_lda))
    print(classification_report(y_test, y_pred_smote_upsampled_lda))
    print('Accuracy:',accuracy_score(y_test, y_pred_smote_upsampled_lda))
    print('Precision:',precision_score(y_test, y_pred_smote_upsampled_lda, average='weighted'))
    print('Recall:',recall_score(y_test, y_pred_smote_upsampled_lda, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_pred_smote_upsampled_lda, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_smote_upsampled_lda))
    pickle.dump(y_pred_smote_upsampled_lda, open('y_pred_smote_upsampled_lda.pkl', 'wb'))

def decisionTreeWithNearMissDownsampledData(x_train_dres_nm,y_train_dres_nm,x_test,y_test):
    dt_classifier_nm_downsampled = tree.DecisionTreeClassifier(criterion='gini')
    dt_classifier_nm_downsampled.fit(x_train_dres_nm,y_train_dres_nm)
    #training prediction
    y_train_pred = dt_classifier_nm_downsampled.predict(x_train_dres_nm)
    #testing prediction
    y_prediction_nm_downsampled = dt_classifier_nm_downsampled.predict(x_test)
    print('f1 score:',f1_score(y_train_dres_nm, y_train_pred, average='weighted'))
    print('Analysis of Decision Tree Algorithm with Near Miss Downsampled Data:\n')
    print(confusion_matrix(y_test, y_prediction_nm_downsampled))
    print(classification_report(y_test, y_prediction_nm_downsampled))
    print('Accuracy:',accuracy_score(y_test, y_prediction_nm_downsampled))
    print('Precision:',precision_score(y_test, y_prediction_nm_downsampled, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction_nm_downsampled, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction_nm_downsampled, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction_nm_downsampled))
    pickle.dump(dt_classifier_nm_downsampled, open('dt_classifier_nm_downsampled.pkl', 'wb'))

def decisionTreeWithPCAandNearMissDownsampledData(x_train_downlsampled_pca,y_train_dres_nm,x_test_downsampled_transformed_pca,y_test):
    dt_classifier_nm_downsampled_pca = tree.DecisionTreeClassifier(criterion='gini')
    dt_classifier_nm_downsampled_pca.fit(x_train_downlsampled_pca,y_train_dres_nm)
    #training prediction
    y_train_pred = dt_classifier_nm_downsampled_pca.predict(x_train_downlsampled_pca)
    #testing prediction
    y_prediction_nm_downsampled_pca = dt_classifier_nm_downsampled_pca.predict(x_test_downsampled_transformed_pca)
    print('F1 Score:',f1_score(y_train_dres_nm, y_train_pred, average='weighted'))
    print('Analysis of Decision Tree Algorithm with PCA and Near Miss Downsampled Data:\n')
    print(confusion_matrix(y_test, y_prediction_nm_downsampled_pca))
    print(confusion_matrix(y_test, y_prediction_nm_downsampled_pca))
    print(classification_report(y_test, y_prediction_nm_downsampled_pca))
    print('Accuracy:',accuracy_score(y_test, y_prediction_nm_downsampled_pca))
    print('Precision:',precision_score(y_test, y_prediction_nm_downsampled_pca, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction_nm_downsampled_pca, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction_nm_downsampled_pca, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction_nm_downsampled_pca))
    pickle.dump(dt_classifier_nm_downsampled_pca, open('dt_classifier_nm_downsampled_pca.pkl', 'wb'))

def decisionTreewithLDAandNearMissDownsampledData(X_train_downsampled_nm_LDA,y_train_dres_nm,X_test_downsampled_nm_LDA_transformed,y_test):
    dt_classifier_nm_downsampled_lda = tree.DecisionTreeClassifier(criterion='gini')
    dt_classifier_nm_downsampled_lda.fit(X_train_downsampled_nm_LDA,y_train_dres_nm)
    #training prediction
    y_train_pred = dt_classifier_nm_downsampled_lda.predict(X_train_downsampled_nm_LDA)
    #testing prediction
    y_prediction_nm_downsampled_lda = dt_classifier_nm_downsampled_lda.predict(X_test_downsampled_nm_LDA_transformed)
    print('F1 Score:',f1_score(y_train_dres_nm, y_train_pred, average='weighted'))
    print('Analysis of Decision Tree Algorithm with LDA and Near Miss Downsampled Data:\n')
    print(confusion_matrix(y_test, y_prediction_nm_downsampled_lda))
    print(confusion_matrix(y_test, y_prediction_nm_downsampled_lda))
    print(classification_report(y_test, y_prediction_nm_downsampled_lda))
    print('Accuracy:',accuracy_score(y_test, y_prediction_nm_downsampled_lda))
    print('Precision:',precision_score(y_test, y_prediction_nm_downsampled_lda, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction_nm_downsampled_lda, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction_nm_downsampled_lda, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction_nm_downsampled_lda))
    pickle.dump(dt_classifier_nm_downsampled_lda, open('dt_classifier_nm_downsampled_lda.pkl', 'wb'))

def decisionTreeWithoutResampledData(x_train,y_train,x_test,y_test):
    dt_classifier_no_resampling = tree.DecisionTreeClassifier(criterion='gini')
    dt_classifier_no_resampling.fit(x_train,y_train)
    #training prediction
    y_train_pred = dt_classifier_no_resampling.predict(x_train)
    #testing prediction
    y_prediction_dt_no_resampling = dt_classifier_no_resampling.predict(x_test)
    print('F1 Score:',f1_score(y_train, y_train_pred, average='weighted'))
    print('Analysis of Decision Tree Algorithm without resampling of Data:\n')
    print(confusion_matrix(y_test, y_prediction_dt_no_resampling))
    print(classification_report(y_test, y_prediction_dt_no_resampling))
    print('Accuracy:',accuracy_score(y_test, y_prediction_dt_no_resampling))
    print('Precision:',precision_score(y_test, y_prediction_dt_no_resampling, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction_dt_no_resampling, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction_dt_no_resampling, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction_dt_no_resampling))
    return dt_classifier_no_resampling

def decisionTreeWithoutResamplingHyperParameterTuning(x_train,y_train,dt_classifier_no_resampling):
   
    dt_classifier_no_resampling.get_depth()
    params = {'max_depth': [40, 50, 60, 70, 80],'max_features':['log2','sqrt',None]}
    gridSearch = GridSearchCV(estimator=dt_classifier_no_resampling, param_grid=params, cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")
    gridSearch.fit(x_train,y_train)
    scoreDF = pd.DataFrame(gridSearch.cv_results_)
    scoreDF.nlargest(5,"mean_test_score")

def decisionTreeWithTuning(x_train,y_train,x_test,y_test):
    
    dt_classifier_no_resampling = tree.DecisionTreeClassifier(criterion='gini', random_state=1,max_depth=70)
    dt_classifier_no_resampling.fit(x_train,y_train)
    #testing prediction
    y_pred_dt_no_resampling_with_hyperparameters = dt_classifier_no_resampling.predict(x_test)

    #training prediction
    y_train_pred = dt_classifier_no_resampling.predict(x_train)
    print('F1 Score:',f1_score(y_train, y_train_pred, average='weighted'))
    print('Analysis of Decision Tree Algorithm without resampling of Data - Setting the right hyperparameters:\n')
    print(confusion_matrix(y_test, y_pred_dt_no_resampling_with_hyperparameters))
    print(classification_report(y_test, y_pred_dt_no_resampling_with_hyperparameters))
    print('Accuracy:',accuracy_score(y_test, y_pred_dt_no_resampling_with_hyperparameters))
    print('Precision:',precision_score(y_test, y_pred_dt_no_resampling_with_hyperparameters, average='weighted'))
    print('Recall:',recall_score(y_test, y_pred_dt_no_resampling_with_hyperparameters, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_pred_dt_no_resampling_with_hyperparameters, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_dt_no_resampling_with_hyperparameters))
    pickle.dump(dt_classifier_no_resampling, open('dt_classifier_no_resampling.pkl', 'wb'))


   