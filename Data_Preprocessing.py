import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class Data_Preprocessor():

    def __init__(self, file_name: str):


        self.df = pd.read_csv(file_name)

        self.cols_with_outliers_to_remove =['ExternalRiskEstimate', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
                               'NumTradesOpeninLast12M', 'NumTradesOpeninLast12M',
                               'MSinceMostRecentInqexcl7days', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance',
                               'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
        
        self.cols_to_handle =[i for i in self.df.columns if i not in self.cols_with_outliers_to_remove]

        self.cols_for_log_transformation = [
        'MSinceMostRecentTradeOpen',
        'PercentTradesNeverDelq',
        'NumInqLast6M',
        'NumInqLast6Mexcl7days',
        'NetFractionRevolvingBurden']


        self.df = self._remove_outliers(self.cols_with_outliers_to_remove)
        self.df = self._handle_outliers(self.cols_to_handle)
        self.df = self._log_transformation()



    def _remove_outliers(self, columns):
        '''This function delete outliers from specified columns
        :param df: the dataframe
        :param columns: columns from which the outliers will be removed'''
        for column in columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]


            self.df = self.df[~((self.df[column] < lower_bound) | (self.df[column] > upper_bound))]

        return self.df
    
    def _handle_outliers(self, columns):
        '''This function replaces outliers with the median value in a column
        :param df: the dataframe
        :param columns: columns where outliers will be replaced'''
        for column in columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]

            self.df[column] = np.where(
                (self.df[column] < lower_bound) | (self.df[column] > upper_bound),
                self.df[column].median(),
                self.df[column]
            )


        return self.df
    
    def _log_transformation(self):

        for col in self.cols_for_log_transformation:
            self.df[col] = np.log1p(self.df[col])
        
        return self.df
    
    

