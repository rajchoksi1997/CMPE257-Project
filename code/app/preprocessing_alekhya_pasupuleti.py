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
import preprocessing_mohamed_shafeeq

def alekhya_preprocessing():
	dataSet = preprocessing_mohamed_shafeeq.shafeeq_preprocessing()
	dataSet = dataSet[(dataSet['PRECTOT'] <= dataSet['PRECTOT'].mean() + 3*dataSet['PRECTOT'].std()) &
        (dataSet['PRECTOT'] >= dataSet['PRECTOT'].mean() - 3*dataSet['PRECTOT'].std())]
	dataSet = dataSet[(dataSet['PS'] <= dataSet['PS'].mean() + 3*dataSet['PS'].std()) &
        (dataSet['PS'] >= dataSet['PS'].mean() - 3*dataSet['PS'].std())]
	dataSet = dataSet[(dataSet['QV2M'] <= dataSet['QV2M'].mean() + 3*dataSet['QV2M'].std()) &
        (dataSet['QV2M'] >= dataSet['QV2M'].mean() - 3*dataSet['QV2M'].std())]
	dataSet = dataSet[(dataSet['T2M'] <= dataSet['T2M'].mean() + 3*dataSet['T2M'].std()) &
        (dataSet['T2M'] >= dataSet['T2M'].mean() - 3*dataSet['T2M'].std())]
	dataSet = dataSet[(dataSet['T2MDEW'] <= dataSet['T2MDEW'].mean() + 3*dataSet['T2MDEW'].std()) &
        (dataSet['T2MDEW'] >= dataSet['T2MDEW'].mean() - 3*dataSet['T2MDEW'].std())]
	dataSet = dataSet[(dataSet['T2MWET'] <= dataSet['T2MWET'].mean() + 3*dataSet['T2MWET'].std()) &
        (dataSet['T2MWET'] >= dataSet['T2MWET'].mean() - 3*dataSet['T2MWET'].std())]
	dataSet = dataSet[(dataSet['T2M_MAX'] <= dataSet['T2M_MAX'].mean() + 3*dataSet['T2M_MAX'].std()) &
        (dataSet['T2M_MAX'] >= dataSet['T2M_MAX'].mean() - 3*dataSet['T2M_MAX'].std())]
	dataSet = dataSet[(dataSet['T2M_MIN'] <= dataSet['T2M_MIN'].mean() + 3*dataSet['T2M_MIN'].std()) &
        (dataSet['T2M_MIN'] >= dataSet['T2M_MIN'].mean() - 3*dataSet['T2M_MIN'].std())]
	dataSet = dataSet[(dataSet['T2M_RANGE'] <= dataSet['T2M_RANGE'].mean() + 3*dataSet['T2M_RANGE'].std()) &
        (dataSet['T2M_RANGE'] >= dataSet['T2M_RANGE'].mean() - 3*dataSet['T2M_RANGE'].std())]
	dataSet = dataSet[(dataSet['TS'] <= dataSet['TS'].mean() + 3*dataSet['TS'].std()) &
        (dataSet['TS'] >= dataSet['TS'].mean() - 3*dataSet['TS'].std())]
	dataSet = dataSet[(dataSet['WS10M'] <= dataSet['WS10M'].mean() + 3*dataSet['WS10M'].std()) &
        (dataSet['WS10M'] >= dataSet['WS10M'].mean() - 3*dataSet['WS10M'].std())]
	dataSet = dataSet[(dataSet['WS10M_MAX'] <= dataSet['WS10M_MAX'].mean() + 3*dataSet['WS10M_MAX'].std()) &
        (dataSet['WS10M_MAX'] >= dataSet['WS10M_MAX'].mean() - 3*dataSet['WS10M_MAX'].std())]
	dataSet = dataSet[(dataSet['WS10M_MIN'] <= dataSet['WS10M_MIN'].mean() + 3*dataSet['WS10M_MIN'].std()) &
        (dataSet['WS10M_MIN'] >= dataSet['WS10M_MIN'].mean() - 3*dataSet['WS10M_MIN'].std())]
	dataSet = dataSet[(dataSet['WS10M_RANGE'] <= dataSet['WS10M_RANGE'].mean() + 3*dataSet['WS10M_RANGE'].std()) &
        (dataSet['WS10M_RANGE'] >= dataSet['WS10M_RANGE'].mean() - 3*dataSet['WS10M_RANGE'].std())]
	dataSet = dataSet[(dataSet['WS50M'] <= dataSet['WS50M'].mean() + 3*dataSet['WS50M'].std()) &
        (dataSet['WS50M'] >= dataSet['WS50M'].mean() - 3*dataSet['WS50M'].std())]
	dataSet = dataSet[(dataSet['WS50M_MAX'] <= dataSet['WS50M_MAX'].mean() + 3*dataSet['WS50M_MAX'].std()) &
        (dataSet['WS50M_MAX'] >= dataSet['WS50M_MAX'].mean() - 3*dataSet['WS50M_MAX'].std())]
	dataSet = dataSet[(dataSet['WS50M_MIN'] <= dataSet['WS50M_MIN'].mean() + 3*dataSet['WS50M_MIN'].std()) &
        (dataSet['WS50M_MIN'] >= dataSet['WS50M_MIN'].mean() - 3*dataSet['WS50M_MIN'].std())]
	dataSet = dataSet[(dataSet['WS50M_RANGE'] <= dataSet['WS50M_RANGE'].mean() + 3*dataSet['WS50M_RANGE'].std()) &
        (dataSet['WS50M_RANGE'] >= dataSet['WS50M_RANGE'].mean() - 3*dataSet['WS50M_RANGE'].std())]


	features = dataSet.drop('score', 1)
	features = features.drop('date', 1)
	features = features.drop('fips', 1)
	
	target = dataSet['score']


	x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)

	return x_train, x_test, y_train, y_test, features, target
	



	