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
from sklearn.neighbors import KNeighborsClassifier

def knnWithoutResampling(x_train,y_train,x_test,y_test):
    knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn_classifier.fit(x_train, y_train)
    y_train_pred = knn_classifier.predict(x_train)
    y_pred_knn = knn_classifier.predict(x_test)
    print('F1 Score:',f1_score(y_train, y_train_pred, average='weighted'))
    print('Performance of KNN Algorithm without resampling:\n')
    print(confusion_matrix(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn))
    print('Accuracy:',accuracy_score(y_test, y_pred_knn))
    print('Precision:',precision_score(y_test, y_pred_knn, average='weighted'))
    print('Recall:',recall_score(y_test, y_pred_knn, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_pred_knn, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_knn))
    pickle.dump(knn_classifier, open('knn_classifier_no_resampling.pkl', 'wb'))
    return knn_classifier

def knnWithoutResamplingHyperParameterTuning(x_train,y_train,knn_classifier):
    k_range = list(range(1, 10))
    param_grid = dict(n_neighbors=k_range)

    grid = GridSearchCV(knn_classifier, param_grid, cv=3, scoring='accuracy', return_train_score=False,verbose=1)
    grid_search=grid.fit(x_train, y_train)
    score_df = pd.DataFrame(grid_search.cv_results_)
    score_df.nlargest(5,"mean_test_score")

def knnWithoutResamplingRightParameters(x_train,y_train,x_test,y_test):
    knn_classifier_right_Parameters = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn_classifier_right_Parameters.fit(x_train, y_train)
    y_train_pred = knn_classifier_right_Parameters.predict(x_train)
    y_pred_knn = knn_classifier_right_Parameters.predict(x_test)
    print('F1 Score:',f1_score(y_train, y_train_pred, average='weighted'))
    print('Performance of KNN Algorithm without resampling - After Hyperparameter Tuning:\n')
    print(confusion_matrix(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn))
    print('Accuracy:',accuracy_score(y_test, y_pred_knn))
    print('Precision:',precision_score(y_test, y_pred_knn, average='weighted'))
    print('Recall:',recall_score(y_test, y_pred_knn, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_pred_knn, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_knn))
    pickle.dump(knn_classifier_right_Parameters, open('knn_classifier_right_Parameters.pkl', 'wb'))

def knnWithUpsampledData(x_train_ures_SMOTE,y_train_ures_SMOTE,x_test,y_test):
    knn_classifier_SMOTE = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn_classifier_SMOTE.fit(x_train_ures_SMOTE, y_train_ures_SMOTE)
    y_train_pred = knn_classifier_SMOTE.predict(x_train_ures_SMOTE)
    y_pred_knn_SMOTE = knn_classifier_SMOTE.predict(x_test)
    print('training f1 score:',f1_score(y_train_ures_SMOTE, y_train_pred, average='weighted'))
    print('Performance of KNN Algorithm with SMOTE Upsampling:\n')
    print(confusion_matrix(y_test, y_pred_knn_SMOTE))
    print(classification_report(y_test, y_pred_knn_SMOTE))
    print('Accuracy:',accuracy_score(y_test, y_pred_knn_SMOTE))
    print('Precision:',precision_score(y_test, y_pred_knn_SMOTE, average='weighted'))
    print('Recall:',recall_score(y_test, y_pred_knn_SMOTE, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_pred_knn_SMOTE, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_knn_SMOTE))
    pickle.dump(knn_classifier_SMOTE, open('knn_classifier_SMOTE_upsampled.pkl', 'wb'))

def knnWithNearDownsampledData(x_train_dres_nm,y_train_dres_nm,x_test,y_test):
    knn_classifier_NM = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn_classifier_NM.fit(x_train_dres_nm, y_train_dres_nm)
    y_pred_knn_NM = knn_classifier_NM.predict(x_test)
    y_train_pred = knn_classifier_NM.predict(x_train_dres_nm)
    print('f1 score:',f1_score(y_train_dres_nm, y_train_pred, average='weighted'))
    print('Performance of KNN Algorithm with NM Downsampling:\n')
    print(confusion_matrix(y_test, y_pred_knn_NM))
    print(classification_report(y_test, y_pred_knn_NM))
    print('Accuracy:',accuracy_score(y_test, y_pred_knn_NM))
    print('Precision:',precision_score(y_test, y_pred_knn_NM, average='weighted'))
    print('Recall:',recall_score(y_test, y_pred_knn_NM, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_pred_knn_NM, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_knn_NM))
    pickle.dump(knn_classifier_NM, open('knn_classifier_NM_downsampled.pkl', 'wb'))

def knnWithPCAandUpsampledData(X_train_upsampled_pca,y_train_ures_SMOTE,X_test_upsampled_transformed_pca,y_test):
    knn_classifier_smote_upsampling_pca = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn_classifier_smote_upsampling_pca.fit(X_train_upsampled_pca,y_train_ures_SMOTE)
    y_prediction_smote_upsampling_pca = knn_classifier_smote_upsampling_pca.predict(X_test_upsampled_transformed_pca)
    print('Analysis of KNN Algorithm with SMOTE Upsampling and PCA:\n')
    print(confusion_matrix(y_test, y_prediction_smote_upsampling_pca))
    print(confusion_matrix(y_test, y_prediction_smote_upsampling_pca))
    print(classification_report(y_test, y_prediction_smote_upsampling_pca))
    print('Accuracy:',accuracy_score(y_test, y_prediction_smote_upsampling_pca))
    print('Precision:',precision_score(y_test, y_prediction_smote_upsampling_pca, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction_smote_upsampling_pca, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction_smote_upsampling_pca, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction_smote_upsampling_pca))
    pickle.dump(knn_classifier_smote_upsampling_pca, open('knn_classifier_smote_upsampling_pca.pkl', 'wb'))

def knnWWithLDAandUpsampledData(X_train_upsampled_LDA,y_train_ures_SMOTE,X_test_upsampled_LDA_transformed,y_test):
    knn_classifier_smote_upsampled_lda = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn_classifier_smote_upsampled_lda.fit(X_train_upsampled_LDA,y_train_ures_SMOTE)
    y_pred_smote_upsampled_lda = knn_classifier_smote_upsampled_lda.predict(X_test_upsampled_LDA_transformed)
    print('Analysis of KNN Algorithm with SMOTE Upsampling and LDA:\n')
    print(confusion_matrix(y_test, y_pred_smote_upsampled_lda))
    print(confusion_matrix(y_test, y_pred_smote_upsampled_lda))
    print(classification_report(y_test, y_pred_smote_upsampled_lda))
    print('Accuracy:',accuracy_score(y_test, y_pred_smote_upsampled_lda))
    print('Precision:',precision_score(y_test, y_pred_smote_upsampled_lda, average='weighted'))
    print('Recall:',recall_score(y_test, y_pred_smote_upsampled_lda, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_pred_smote_upsampled_lda, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_smote_upsampled_lda))
    pickle.dump(y_pred_smote_upsampled_lda, open('y_pred_smote_upsampled_lda.pkl', 'wb'))

def knnWithLDAandDownsampledData(X_train_downsampled_nm_LDA,y_train_dres_nm,X_test_downsampled_nm_LDA_transformed,y_test):
    knn_classifier_nm_downsampled_lda = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn_classifier_nm_downsampled_lda.fit(X_train_downsampled_nm_LDA,y_train_dres_nm)
    y_prediction_nm_downsampled_lda = knn_classifier_nm_downsampled_lda.predict(X_test_downsampled_nm_LDA_transformed)
    pickle.dump(knn_classifier_nm_downsampled_lda, open('dt_classifier_nm_downsampled_lda.pkl', 'wb'))
    print('Analysis of KNN Algorithm with Near Miss Downsampling and LDA:\n')
    print(confusion_matrix(y_test, y_prediction_nm_downsampled_lda))
    print(confusion_matrix(y_test, y_prediction_nm_downsampled_lda))
    print(classification_report(y_test, y_prediction_nm_downsampled_lda))
    print('Accuracy:',accuracy_score(y_test, y_prediction_nm_downsampled_lda))
    print('Precision:',precision_score(y_test, y_prediction_nm_downsampled_lda, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction_nm_downsampled_lda, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction_nm_downsampled_lda, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction_nm_downsampled_lda))
    pickle.dump(knn_classifier_nm_downsampled_lda, open('knn_classifier_nm_downsampled_lda.pkl', 'wb'))
    
def knnWithPCAandDownsampledData(x_train_downlsampled_pca,y_train_dres_nm,x_test_downsampled_transformed_pca,y_test):
    knn_classifier_nm_downsampled_pca = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    knn_classifier_nm_downsampled_pca.fit(x_train_downlsampled_pca,y_train_dres_nm)
    y_prediction_nm_downsampled_pca = knn_classifier_nm_downsampled_pca.predict(x_test_downsampled_transformed_pca)
    print('Analysis of KNN Algorithm with Near Miss Downsampling and PCA:\n')
    print(confusion_matrix(y_test, y_prediction_nm_downsampled_pca))
    print(confusion_matrix(y_test, y_prediction_nm_downsampled_pca))
    print(classification_report(y_test, y_prediction_nm_downsampled_pca))
    print('Accuracy:',accuracy_score(y_test, y_prediction_nm_downsampled_pca))
    print('Precision:',precision_score(y_test, y_prediction_nm_downsampled_pca, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction_nm_downsampled_pca, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction_nm_downsampled_pca, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction_nm_downsampled_pca))
    pickle.dump(knn_classifier_nm_downsampled_pca, open('knn_classifier_nm_downsampled_pca.pkl', 'wb'))
    pickle.dump(knn_classifier_nm_downsampled_pca, open('knn_classifier_nm_downsampled_pca.pkl', 'wb'))

