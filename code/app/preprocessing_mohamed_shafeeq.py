import pandas as pd

def shafeeq_preprocessing():
	dataSet = pd.read_csv('train_timeseries.csv')
	print("First 5 rows: \n",dataSet.head())

	print("Last 5 rows: \n",dataSet.tail())
	print("Data show Describe: \n",dataSet.describe())
	print("Datatype of Attributes: \n",dataSet.info())
	print("Columns \n",dataSet.columns)
	print("Shape of the DataSet \n",dataSet.shape)
	print("Sum of null values \n",dataSet.isnull().sum())
	print("Are there Null Values in Data\n",dataSet.isnull().values.any())
	# Drop missing values in target variable
	print("Dropping Null Values")
	dataSet = dataSet.dropna()
	print(dataSet.isnull().sum())
	
	date = dataSet['date']
	dataSet['year'] = pd.DatetimeIndex(date).year
	dataSet['month'] = pd.DatetimeIndex(date).month
	dataSet['day'] = pd.DatetimeIndex(date).day
	dataSet['score'] = dataSet['score'].round().astype(int)
	dataSet = removeSpecialCharacter(dataSet)
	dataSet.skew(axis = 0, skipna = True)
	dataSet.kurtosis(axis = 0, skipna = True)

	return dataSet

def removeSpecialCharacter(dataSet):

	for c in ['fips', 'date', 'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET','T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX','WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN','WS50M_RANGE', 'score']:
		unique_val_cols = dataSet[c].unique()
		print('Unique values in ' , c , 'are ', unique_val_cols)

	for c in ['fips', 'date', 'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET','T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX','WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN','WS50M_RANGE', 'score']:
		dataSet[c] = pd.to_numeric(dataSet[c], errors='coerce')
		
	return dataSet
		
	

