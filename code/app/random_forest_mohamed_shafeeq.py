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
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


def randomForestWithSmoteUpsampledData(x_train_ures_SMOTE,y_train_ures_SMOTE,x_test,y_test):

    rf_classifer_ups_smote = RandomForestClassifier(n_estimators = 20, max_depth=70, random_state=0)
    rf_classifer_ups_smote.fit(x_train_ures_SMOTE,y_train_ures_SMOTE)
    # prediction of train data
    rf_ytrain_pred = rf_classifer_ups_smote.predict(x_train_ures_SMOTE)
    # prediction of test data
    rf_ups_smote_y_prediction = rf_classifer_ups_smote.predict(x_test)
    print('training f1 score:',f1_score(y_train_ures_SMOTE, rf_ytrain_pred, average='weighted'))
    print('Results of Random Forest Algorithm with SMOTE upsampled data:\n')
    print(confusion_matrix(y_test, rf_ups_smote_y_prediction))
    print(classification_report(y_test, rf_ups_smote_y_prediction))
    print('Accuracy:',accuracy_score(y_test, rf_ups_smote_y_prediction))
    print('Precision:',precision_score(y_test, rf_ups_smote_y_prediction, average='weighted'))
    print('Recall:',recall_score(y_test, rf_ups_smote_y_prediction, average='weighted'))
    print('F1 Score:',f1_score(y_test, rf_ups_smote_y_prediction, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, rf_ups_smote_y_prediction))
    pickle.dump(rf_ups_smote_y_prediction, open('rf_classifer_upsampling_smote.pkl', 'wb'))

def randomForestPCAandUpsampledData(X_train_upsampled_pca,y_train_ures_SMOTE,X_test_upsampled_transformed_pca,y_test):
    rf_classifer_ups_smote_ups_pca = RandomForestClassifier(n_estimators = 20, max_depth=70, random_state=0)
    rf_classifer_ups_smote_ups_pca.fit(X_train_upsampled_pca,y_train_ures_SMOTE)
    # prediction of train data
    rf_ytrain_pred = rf_classifer_ups_smote_ups_pca.predict(X_train_upsampled_pca)
    # prediction of test data
    rf_pca_ups_smote_y_prediction = rf_classifer_ups_smote_ups_pca.predict(X_test_upsampled_transformed_pca)
    print('training f1 score:',f1_score(y_train_ures_SMOTE, rf_ytrain_pred, average='weighted'))
    print('Results of Random Forest Algorithm with PCA and SMOTE upsampled data:\n')
    print(confusion_matrix(y_test, rf_pca_ups_smote_y_prediction))
    print(classification_report(y_test, rf_pca_ups_smote_y_prediction))
    print('Accuracy:',accuracy_score(y_test, rf_pca_ups_smote_y_prediction))
    print('Precision:',precision_score(y_test, rf_pca_ups_smote_y_prediction, average='weighted'))
    print('Recall:',recall_score(y_test, rf_pca_ups_smote_y_prediction, average='weighted'))
    print('F1 Score:',f1_score(y_test, rf_pca_ups_smote_y_prediction, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, rf_pca_ups_smote_y_prediction))
    pickle.dump(rf_pca_ups_smote_y_prediction, open('rf_classifer_smote_upsampling_pca.pkl', 'wb'))

def randomForestWithLDAUpsampledData(X_train_upsampled_LDA,y_train_ures_SMOTE,X_test_upsampled_LDA_transformed,y_test):
    rf_classifer_smote_ups_lda = RandomForestClassifier(n_estimators = 20, max_depth=70, random_state=0)
    rf_classifer_smote_ups_lda.fit(X_train_upsampled_LDA,y_train_ures_SMOTE)
    # prediction of train data
    rf_ytrain_pred = rf_classifer_smote_ups_lda.predict(X_train_upsampled_LDA)
    # prediction of test data
    rf_lda_ups_smote_y_prediction = rf_classifer_smote_ups_lda.predict(X_test_upsampled_LDA_transformed)
    print('F1 Score:',f1_score(y_train_ures_SMOTE, rf_ytrain_pred, average='weighted'))
    print('Results of Random Forest Algorithm with LDA and SMOTE upsampled data:\n')
    print(confusion_matrix(y_test, rf_lda_ups_smote_y_prediction))
    print(classification_report(y_test, rf_lda_ups_smote_y_prediction))
    print('Accuracy:',accuracy_score(y_test, rf_lda_ups_smote_y_prediction))
    print('Precision:',precision_score(y_test, rf_lda_ups_smote_y_prediction, average='weighted'))
    print('Recall:',recall_score(y_test, rf_lda_ups_smote_y_prediction, average='weighted'))
    print('F1 Score:',f1_score(y_test, rf_lda_ups_smote_y_prediction, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, rf_lda_ups_smote_y_prediction))
    pickle.dump(rf_lda_ups_smote_y_prediction, open('rf_classifer_smote_upsampling_lda.pkl', 'wb'))

def randomForestwithDownsampledData(x_train_dres_nm,y_train_dres_nm,x_test,y_test):
    rf_classifer_nm_downsample = RandomForestClassifier(n_estimators = 20, max_depth=70, random_state=0)
    rf_classifer_nm_downsample.fit(x_train_dres_nm,y_train_dres_nm)
    # prediction of train data
    rf_ytrain_pred = rf_classifer_nm_downsample.predict(x_train_dres_nm)
    # prediction of test data
    rf_nm_downsampled_y_prediction = rf_classifer_nm_downsample.predict(x_test)
    print('F1 Score:',f1_score(y_train_dres_nm, rf_ytrain_pred, average='weighted'))
    print('Results of Random Forest Algorithm with Near Miss Downsample Data:\n')
    print(confusion_matrix(y_test, rf_nm_downsampled_y_prediction))
    print(classification_report(y_test, rf_nm_downsampled_y_prediction))
    print('Accuracy:',accuracy_score(y_test, rf_nm_downsampled_y_prediction))
    print('Precision:',precision_score(y_test, rf_nm_downsampled_y_prediction, average='weighted'))
    print('Recall:',recall_score(y_test, rf_nm_downsampled_y_prediction, average='weighted'))
    print('F1 Score:',f1_score(y_test, rf_nm_downsampled_y_prediction, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, rf_nm_downsampled_y_prediction))
    pickle.dump(rf_nm_downsampled_y_prediction, open('rf_classifer_nm_downsample.pkl', 'wb'))
    

def randomForestWithPCAandDownsampledData(x_train_downlsampled_pca,y_train_dres_nm,x_test_downsampled_transformed_pca,y_test,x_train_dres_nm,x_test):
    rf_classifer_nm_downsample_pca = RandomForestClassifier(n_estimators = 20, max_depth=70, random_state=0)
 
    rf_classifer_nm_downsample_pca.fit(x_train_downlsampled_pca,y_train_dres_nm)
    # prediction of train data
    rf_ytrain_pred = rf_classifer_nm_downsample_pca.predict(x_train_downlsampled_pca)
    # prediction of test data
    rf_nm_downsampled_pca_y_prediction = rf_classifer_nm_downsample_pca.predict(x_test_downsampled_transformed_pca)
    print('F1 Score:',f1_score(y_train_dres_nm, rf_ytrain_pred, average='weighted'))
    print('Results of Random Forest Algorithm with PCA and Near Miss Downsampled Data:\n')
    print(confusion_matrix(y_test, rf_nm_downsampled_pca_y_prediction))
    print(confusion_matrix(y_test, rf_nm_downsampled_pca_y_prediction))
    print(classification_report(y_test, rf_nm_downsampled_pca_y_prediction))
    print('Accuracy:',accuracy_score(y_test, rf_nm_downsampled_pca_y_prediction))
    print('Precision:',precision_score(y_test, rf_nm_downsampled_pca_y_prediction, average='weighted'))
    print('Recall:',recall_score(y_test, rf_nm_downsampled_pca_y_prediction, average='weighted'))
    print('F1 Score:',f1_score(y_test, rf_nm_downsampled_pca_y_prediction, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, rf_nm_downsampled_pca_y_prediction))
    pickle.dump(rf_nm_downsampled_pca_y_prediction, open('rf_classifer_nm_downsample_PCA.pkl', 'wb'))

def randomForestWithLDAandDownsampledData(X_train_downsampled_nm_LDA,y_train_dres_nm,X_test_downsampled_nm_LDA_transformed,y_test):
    rf_classifer_nm_downsample_lda = RandomForestClassifier(n_estimators = 20, max_depth=70, random_state=0)
    rf_classifer_nm_downsample_lda.fit(X_train_downsampled_nm_LDA,y_train_dres_nm)
    # prediction of train data
    rf_ytrain_pred = rf_classifer_nm_downsample_lda.predict(X_train_downsampled_nm_LDA)
    # prediction of test data
    rf_nm_downsampled_lda_y_prediction = rf_classifer_nm_downsample_lda.predict(X_test_downsampled_nm_LDA_transformed)
    print('F1 Score:',f1_score(y_train_dres_nm, rf_ytrain_pred, average='weighted'))
    print('Results of Random Forest Algorithm with LDA and Near Miss Downsampled Data:\n')
    print(confusion_matrix(y_test, rf_nm_downsampled_lda_y_prediction))
    print(confusion_matrix(y_test, rf_nm_downsampled_lda_y_prediction))
    print(classification_report(y_test, rf_nm_downsampled_lda_y_prediction))
    print('Accuracy:',accuracy_score(y_test, rf_nm_downsampled_lda_y_prediction))
    print('Precision:',precision_score(y_test, rf_nm_downsampled_lda_y_prediction, average='weighted'))
    print('Recall:',recall_score(y_test, rf_nm_downsampled_lda_y_prediction, average='weighted'))
    print('F1 Score:',f1_score(y_test, rf_nm_downsampled_lda_y_prediction, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, rf_nm_downsampled_lda_y_prediction))
    pickle.dump(rf_nm_downsampled_lda_y_prediction, open('rf_classifer_nm_downsample_LDA.pkl', 'wb'))

def randomForestWithoutResampling(x_train,y_train,x_test,y_test):
    rf_classifer_no_resample = RandomForestClassifier(n_estimators = 20, max_depth=70, random_state=0)
    rf_classifer_no_resample.fit(x_train,y_train)
    # prediction of train data
    rf_ytrain_pred = rf_classifer_no_resample.predict(x_train)
    # prediction of test data
    rf_no_resample_y_prediction = rf_classifer_no_resample.predict(x_test)
    print('F1 Score:',f1_score(y_train, rf_ytrain_pred, average='weighted'))
    print('Results of Random Forest Algorithm without Resampling:\n')
    print(confusion_matrix(y_test, rf_no_resample_y_prediction))
    print(confusion_matrix(y_test, rf_no_resample_y_prediction))
    print(classification_report(y_test, rf_no_resample_y_prediction))
    print('Accuracy:',accuracy_score(y_test, rf_no_resample_y_prediction))
    print('Precision:',precision_score(y_test, rf_no_resample_y_prediction, average='weighted'))
    print('Recall:',recall_score(y_test, rf_no_resample_y_prediction, average='weighted'))
    print('F1 Score:',f1_score(y_test, rf_no_resample_y_prediction, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, rf_no_resample_y_prediction))
    pickle.dump(rf_no_resample_y_prediction, open('rf_classifer_no_resample.pkl', 'wb'))
    return rf_classifer_no_resample

def randomForestWithoutResamplingHyperparaTuning(x_train,y_train,x_test,y_test,rf_classifer_no_resample):
    rf_classifer_no_resample = randomForestWithoutResampling(x_train,y_train,x_test,y_test)

    rf_max_depth = [int(j) for j in np.linspace(10, 110, num = 11)]
    rf_max_depth.append(None)
    rf_grid = {'n_estimators': [int(j) for j in np.linspace(start = 10, stop = 50, num = 10)],'max_features': ['auto', 'sqrt'],'max_depth': rf_max_depth,'bootstrap': [True,False]}
    rf_rand = RandomizedSearchCV(estimator = rf_classifer_no_resample, param_distributions = rf_grid, n_iter = 20, cv = 3, verbose=2, random_state=0, n_jobs = -1)
    rf_rand.fit(x_train, y_train)
    rf_rand.best_params_

def randomForestNoresamplingRightParameters(x_train,y_train,y_test,x_test):
    rf_classifer_no_resample_right_hyp = RandomForestClassifier(n_estimators = 50, max_depth=80, bootstrap=False, max_features='sqrt', random_state=0)
    rf_classifer_no_resample_right_hyp.fit(x_train, y_train)
    rf_no_resample_right_hyp_y_prediction = rf_classifer_no_resample_right_hyp.predict(x_test)
    rf_train_pred = rf_classifer_no_resample_right_hyp.predict(x_train)
    print('Train F1 Score:',f1_score(y_train, rf_train_pred, average='weighted'))
    print('Results of Random Forest Algorithm without Resampling - Right Hyperparameters Modification:\n')
    print(confusion_matrix(y_test, rf_no_resample_right_hyp_y_prediction))
    print(confusion_matrix(y_test, rf_no_resample_right_hyp_y_prediction))
    print(classification_report(y_test, rf_no_resample_right_hyp_y_prediction))
    print('Accuracy:',accuracy_score(y_test, rf_no_resample_right_hyp_y_prediction))
    print('Precision:',precision_score(y_test, rf_no_resample_right_hyp_y_prediction, average='weighted'))
    print('Recall:',recall_score(y_test, rf_no_resample_right_hyp_y_prediction, average='weighted'))
    print('F1 Score:',f1_score(y_test, rf_no_resample_right_hyp_y_prediction, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, rf_no_resample_right_hyp_y_prediction))
    pickle.dump(rf_no_resample_right_hyp_y_prediction, open('rf_classifer_no_resample_right_hyperparameter.pkl', 'wb'))