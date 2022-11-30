from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import preprocessing_piyush_gade

def raj_preprocessing():
	x_train_dres, y_train_dres, x_train_dres_nm, y_train_dres_nm, x_train_ures_SMOTE, y_train_ures_SMOTE, x_test, y_test,x_train,y_train = preprocessing_piyush_gade.piyush_preprocessing()

	LDanalysis = LDA(n_components=5)
	X_train_upsampled_LDA=LDanalysis.fit_transform(x_train_ures_SMOTE,y_train_ures_SMOTE)
	X_test_upsampled_LDA_transformed=LDanalysis.transform(x_test)

	LDanalysis = LDA(n_components=5)
	X_train_downsampled_nm_LDA=LDanalysis.fit_transform(x_train_dres_nm,y_train_dres_nm)
	X_test_downsampled_nm_LDA_transformed=LDanalysis.transform(x_test)

	PCanalysis = PCA(n_components=5)
	X_train_upsampled_pca = PCanalysis.fit_transform(x_train_ures_SMOTE)
	X_test_upsampled_transformed_pca = PCanalysis.transform(x_test)

	PCanalysis = PCA()
	x_train_downlsampled_pca = PCanalysis.fit_transform(x_train_dres_nm)
	x_test_downsampled_transformed_pca = PCanalysis.transform(x_test)

	PCanalysis = PCA(n_components = 5)
	x_train_downlsampled_pca = PCanalysis.fit_transform(x_train_dres_nm)
	x_test_downsampled_transformed_pca = PCanalysis.transform(x_test)
	
	return X_train_upsampled_LDA, X_test_upsampled_LDA_transformed, X_train_downsampled_nm_LDA, X_test_downsampled_nm_LDA_transformed, X_train_upsampled_pca, X_test_upsampled_transformed_pca, x_train_downlsampled_pca, x_test_downsampled_transformed_pca, x_train_downlsampled_pca, x_test_downsampled_transformed_pca,x_train_dres, y_train_dres, x_train_dres_nm, y_train_dres_nm, x_train_ures_SMOTE, y_train_ures_SMOTE, x_test, y_test,x_train,y_train

