import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')


class Data_Cleaner():

    def __init__(self, file_name: str):
        # --- Retrieves the data from CSV file ---
        #Load dataset
        self.df = pd.read_csv(file_name)
        self.dict_of_auc_score = {}
        self.dict_of_imputation = {}
        
        #Mapping categorical values to numerical ones
        self.df['RiskPerformance'] = self.df['RiskPerformance'].map({'Bad':0, 'Good':1})
        self.df = self._replace_spec_vals()
        self.df = self._aggregate_columns()
        self.df = self._imputing()



    def _replace_spec_vals(self):
            '''This function handle specific values (-7, -8, -9) it replaces them with nan value
                :param df: the dataframe that contains specific values to handle'''

            spec_vals = [-7, -8, -9]
            for val in spec_vals:
                self.df.replace(val, np.nan, inplace=True)

            return self.df
    
    def _aggregate_columns(self):
         
        #Columns MaxDelqEver and MaxDelq2PublicRecLast12M contains values that could bias the model 
        #so we extract from these columns new features which could more illustrative
        self.df['DeragatoryComment'] = (self.df['MaxDelq2PublicRecLast12M'] == 0).astype(int)
        self.df['120Delinquent'] = (self.df['MaxDelq2PublicRecLast12M'] == 1).astype(int)
        self.df['90Delinquent'] = (self.df['MaxDelq2PublicRecLast12M'] == 2).astype(int)
        self.df['60Delinquent'] = (self.df['MaxDelq2PublicRecLast12M'] == 3).astype(int)
        self.df['30Delinquent'] = (self.df['MaxDelq2PublicRecLast12M'] == 4).astype(int)
        self.df['UnkownDelinquency'] = ((self.df['MaxDelq2PublicRecLast12M'] == 5) | (self.df['MaxDelq2PublicRecLast12M'] == 6)).astype(int)
        self.df['NeverDelinquent'] = (self.df['MaxDelq2PublicRecLast12M'] == 7).astype(int)

        for row in range(self.df.shape[0]):
            if (self.df['MaxDelqEver'][row] == 2 & self.df['DeragatoryComment'][row] == 0):
                self.df['DeragatoryComment'][row] += 1
            elif (self.df['MaxDelqEver'][row] == 3 & self.df['120Delinquent'][row] == 0):
                self.df['120Delinquent'][row] += 1
            elif (self.df['MaxDelqEver'][row] == 4 & self.df['90Delinquent'][row] == 0):
                self.df['90Delinquent'][row] += 1
            elif (self.df['MaxDelqEver'][row] == 5 & self.df['60Delinquent'][row] == 0):
                self.df['60Delinquent'][row] += 1
            elif (self.df['MaxDelqEver'][row] == 6 & self.df['60Delinquent'][row] == 0):
                self.df['60Delinquent'][row] += 1
            elif (self.df['MaxDelqEver'][row] == 7 & self.df['UnkownDelinquency'][row] == 0):
                self.df['UnkownDelinquency'][row] += 1
            elif (self.df['MaxDelqEver'][row] == 8 & self.df['NeverDelinquent'][row] == 0):
                self.df['NeverDelinquent'][row] += 1
        
        self.df = self.df.drop(columns=['MaxDelqEver', 'MaxDelq2PublicRecLast12M'])

        return self.df
    
    

    def _check_imputation_performance(self, df, imputation_strategy):
        '''This function imputes the nan values in a dataframe and checks its performance
        :param imputation_strategy: the imputation strategy to test its performance'''
        
        model = RandomForestClassifier()
        y = df['RiskPerformance']
        X = df.drop(columns=['RiskPerformance'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        self.dict_of_auc_score[imputation_strategy] = roc_auc

        return self.dict_of_auc_score
    
    def _imputing(self):


        df_mean_imputed = self.df.fillna(self.df.mean())

        df_median_imputed = self.df.fillna(self.df.median())

        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(self.df)
        df_KNN_imputed = pd.DataFrame(imputed_data, columns=self.df.columns)

        imputer = IterativeImputer(max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(self.df)
        df_Iterative_imputed = pd.DataFrame(imputed_data, columns=self.df.columns)

        # Linear Interpolation
        df_linear_imputed = self.df.interpolate(method='linear')

        # Polynomial Interpolation
        df_polynomial_imputed = self.df.interpolate(method='polynomial', order=2)

        self.dict_of_imputation['Mean Imputation'] = df_mean_imputed
        self.dict_of_imputation['Median Imputation'] = df_median_imputed
        self.dict_of_imputation['KNN Imputation'] = df_KNN_imputed
        self.dict_of_imputation['Iterative Imputation'] = df_Iterative_imputed
        self.dict_of_imputation['Linear Imputation'] = df_linear_imputed
        self.dict_of_imputation['Polynomial Imputation'] = df_polynomial_imputed

        self.dict_of_auc_score = self._check_imputation_performance(df_mean_imputed, 'Mean Imputation')
        self.dict_of_auc_score = self._check_imputation_performance(df_median_imputed, 'Median Imputation')
        self.dict_of_auc_score = self._check_imputation_performance(df_KNN_imputed, 'KNN Imputation')
        self.dict_of_auc_score = self._check_imputation_performance(df_Iterative_imputed, 'Iterative Imputation')
        self.dict_of_auc_score = self._check_imputation_performance(df_linear_imputed, 'Linear Imputation')
        self.dict_of_auc_score = self._check_imputation_performance(df_polynomial_imputed, 'Polynomial Imputation')


        best_strategy = max(self.dict_of_auc_score, key=self.dict_of_auc_score.get)
        self.df = self.dict_of_imputation[best_strategy]

        return self.df
    
    def output_to_csv(self, file_name: str):

        # Save DataFrame to a CSV file
        self.df.to_csv(file_name, index=False)


    
      
