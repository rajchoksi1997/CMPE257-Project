import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import NearMiss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, KernelPCA
import preprocessing_alekhya_pasupuleti
def piyush_preprocessing():
	x_train, x_test, y_train, y_test, features, target = preprocessing_alekhya_pasupuleti.alekhya_preprocessing()
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)

	Estimator = RandomForestClassifier(n_estimators=9)
	FeatureExtractor = RFE(estimator=Estimator, n_features_to_select=15)
	fitter = FeatureExtractor.fit(x_train, y_train)
	extracted_features = features.columns[(fitter.get_support())]


	features = features.drop('PRECTOT', 1)
	features = features.drop('T2MWET', 1)
	features = features.drop('WS10M_MAX', 1)
	features = features.drop('WS10M_MIN', 1)
	features = features.drop('WS50M_MIN', 1)
	features = features.drop('month', 1)
	
	x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
	

	sc = StandardScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.transform(x_test)
	

	downsampler = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
	x_train_dres, y_train_dres = downsampler.fit_resample(x_train, y_train)

	downsampler = NearMiss()
	x_train_dres_nm, y_train_dres_nm = downsampler.fit_resample(x_train, y_train)

	upsampler = SMOTE(random_state = 5)
	x_train_ures_SMOTE, y_train_ures_SMOTE = upsampler.fit_resample(x_train, y_train.ravel())
	
	return x_train_dres, y_train_dres, x_train_dres_nm, y_train_dres_nm, x_train_ures_SMOTE, y_train_ures_SMOTE, x_test, y_test,x_train,y_train