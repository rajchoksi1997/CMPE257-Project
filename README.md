###Team 5
### Raj Bharatbhai Choksi - [rajchoksi1997](https://github.com/rajchoksi1997)
### Piyush Gade - [Piyush1729](https://github.com/Piyush1729)
### Alekhya Pasupuleti - [AlekhyaPasupuleti](https://github.com/AlekhyaPasupuleti)
### Mohamed Shafeeq Usman - [MdShafeeqU](https://github.com/MdShafeeqU)

# Comparative Study of Machine Learning models in Drought Prediction

##Problem Statement:
Predicting drought and its severity using various machine learning models and comparing those results.

##Abstract
Agriculture is an important part of the US economy. According to the US government agriculture contributed $1.5 trillion to the economy in 2020 which is a 5% share. However global warming and changes in climate leads to significant drought in various parts of the country which adversely affects agriculture. Unlike other natural disasters, Drought develops slowly and has long term consequences. Hence by leveraging machine learning we can help farmers in taking preventive measures and minimize their loss. 

Our aim is to provide a comparative study on the performance of different machine learning models in predicting five levels of drought ranging from moderate to extreme using meteorological data. Weather conditions and precipitation levels at different heights from the sea level will play an important indicators for predicting droughts. We aim to use supervised learning models such as Random Forest, Decision Tree, KNN and naive bayes for the study and compare their results using performance metrics such as F1 score, accuracy, recall, precision and ROC curve.  

We will be using dataset from the [US drought monitor](https://droughtmonitor.unl.edu/DmData/DataDownload.aspx) which provides drought data and meteorological statistics from year 2000 onwards. The dataset is updated on a weekly basis. As a part of preprocessing of the dataset we dropped the null values and removed special characters from numerical columns which left us with approximately three millions rows in our dataset. Additionally, There are twenty feature and one target variable which gives score related to the severity of drought. Out of the twenty features one is categorical and the rest are continuous. The only categorical feature is the date hence we transformed it into three numerical features namely day, year and month. We featured binned target variable into six classes and plotted histogram for all the features Additionally we performed Univariate and bivariate analysis. Lastly we understood the correlation between independent variables.

##Insights and plots after preprocessing of dataset are included in the DroughtPrediction.pdf file.


