from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, KernelPCA
import pickle

def summariseCV(grid_result):
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def metrics(y_test, y_prediction):
    print(confusion_matrix(y_test, y_prediction))
    print(confusion_matrix(y_test, y_prediction))
    print(classification_report(y_test, y_prediction))
    print('Accuracy:',accuracy_score(y_test, y_prediction))
    print('Precision:',precision_score(y_test, y_prediction, average='weighted'))
    print('Recall:',recall_score(y_test, y_prediction, average='weighted'))
    print('F1 Score:',f1_score(y_test, y_prediction, average='weighted'))
    print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_prediction))

def roc(y_test, y_prediction):
    falsePositiveRate = dict()
    truePositiveRate = dict()
    threshold = dict()

    for i in range(6):    
        falsePositiveRate[i], truePositiveRate[i], threshold[i] = roc_curve(y_test, y_prediction, pos_label=i)

    plt.plot(falsePositiveRate[0], truePositiveRate[0], linestyle='--',color='blue', label='Class 0 vs Rest')
    plt.plot(falsePositiveRate[1], truePositiveRate[1], linestyle='--',color='yellow', label='Class 1 vs Rest')
    plt.plot(falsePositiveRate[2], truePositiveRate[2], linestyle='--',color='orangered', label='Class 2 vs Rest')
    plt.plot(falsePositiveRate[3], truePositiveRate[3], linestyle='--',color='green', label='Class 3 vs Rest')
    plt.plot(falsePositiveRate[4], truePositiveRate[4], linestyle='--',color='magenta', label='Class 4 vs Rest')
    plt.plot(falsePositiveRate[5], truePositiveRate[5], linestyle='--',color='purple', label='Class 5 vs Rest')

    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()
    

def logisticRegression(x_train, y_train, x_test, y_test):
    model = LogisticRegression(multi_class='multinomial')
    solvers = ['newton-cg', 'lbfgs', 'sag', 'saga' ]
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1_macro', error_score=0)
    grid_result = grid_search.fit(x_train, y_train)
    y_prediction = grid_search.predict(x_test)
    summariseCV(grid_result)
    metrics(y_test, y_prediction)
    roc(y_test, y_prediction)

#PCA on NCR downsample
def pcaNcr(x_train_dres, x_test):
    PCanalysis = PCA(n_components = 5)
    # picking values with more than 90% variance
    x_train_ncr_pca = PCanalysis.fit_transform(x_train_dres)
    x_test_ncr_transformed_pca = PCanalysis.transform(x_test)
    print("Variance Ratio: " ,PCanalysis.explained_variance_ratio_)
    return x_train_ncr_pca, x_test_ncr_transformed_pca

#LDA on NCR downsample
def ldaNcr(x_train_dres, y_train_dres, x_test):
    LDanalysis = LDA(n_components=5)
    X_train_downsampled_ncr_LDA=LDanalysis.fit_transform(x_train_dres, y_train_dres)
    X_test_downsampled_ncr_LDA_transformed=LDanalysis.transform(x_test)
    return X_train_downsampled_ncr_LDA, X_test_downsampled_ncr_LDA_transformed

#near miss
def logisticRegressionNm(x_train_dres_nm, y_train_dres_nm, x_test, y_test):
    logisticRegression(x_train_dres_nm, y_train_dres_nm, x_test, y_test)

def logisticRegressionNmLda(X_train_downsampled_nm_LDA, y_train_dres_nm, X_test_downsampled_nm_LDA_transformed, y_test):
    logisticRegression(X_train_downsampled_nm_LDA, y_train_dres_nm, X_test_downsampled_nm_LDA_transformed, y_test)

def logisticRegressionNmPca(x_train_downlsampled_pca, y_train_dres_nm, x_test_downsampled_transformed_pca, y_test):
    logisticRegression(x_train_downlsampled_pca, y_train_dres_nm, x_test_downsampled_transformed_pca, y_test)

#upsample
def logisticRegressionSmote(x_train_ures_SMOTE, y_train_ures_SMOTE, x_test, y_test):
    logisticRegression(x_train_ures_SMOTE, y_train_ures_SMOTE, x_test, y_test)

def logisticRegressionSmoteLda(X_train_upsampled_LDA, y_train_ures_SMOTE, X_test_upsampled_LDA_transformed, y_test):
    logisticRegression(X_train_upsampled_LDA, y_train_ures_SMOTE, X_test_upsampled_LDA_transformed, y_test)

def logisticRegressionSmotePca(X_train_upsampled_pca, y_train_ures_SMOTE, X_test_upsampled_transformed_pca, y_test):
    logisticRegression(X_train_upsampled_pca, y_train_ures_SMOTE, X_test_upsampled_transformed_pca, y_test)

#NCR
def logisticRegressionNcr(x_train_dres, y_train_dres, x_test, y_test):
    logisticRegression(x_train_dres, y_train_dres, x_test, y_test)

def logisticRegressionNcrLda(x_train_dres, y_train_dres, x_test, y_test):
    X_train_downsampled_ncr_LDA, X_test_downsampled_ncr_LDA_transformed = ldaNcr(x_train_dres, y_train_dres, x_test)
    logisticRegression(X_train_downsampled_ncr_LDA, y_train_dres, X_test_downsampled_ncr_LDA_transformed, y_test)

def logisticRegressionNcrPca(x_train_dres, y_train_dres, x_test, y_test):
    x_train_ncr_pca, x_test_ncr_transformed_pca = pcaNcr(x_train_dres, x_test)
    logisticRegression(x_train_ncr_pca, y_train_dres, x_test_ncr_transformed_pca, y_test)

#no resampling
def logisticRegressionNres(x_train, y_train, x_test, y_test):
    logisticRegression(x_train, y_train, x_test, y_test)