import sys
import pandas as pd
from datetime import datetime
# ############################################################################################################################
from sklearn.model_selection import train_test_split
# #############################################################################################################################
from model_combination import make_voting_classifier
from result_check_tool import voting_compare_models_result
from ml_models import models_dict_make
from sampling_methods import sampling_dict_make
#############################################################################################################################
import warnings
warnings.filterwarnings('ignore')


models_dict = models_dict_make()
sampling_dict = sampling_dict_make()

train_data_user_encoded = pd.read_csv('C:/py_workspace/jupyter/train_data_user_encoded_mean_stand.csv', index_col='key')
predict_data_user_encoded = pd.read_csv('C:/py_workspace/jupyter/predict_data_user_encoded.csv', index_col='key')


X = train_data_user_encoded.drop('Repurchase', axis = 1)
y = train_data_user_encoded['Repurchase']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)



for name, method in sampling_dict.items():

    sys.stdout = open('{0}_{1}.txt'.format(datetime.today().strftime("%Y_%m_%d_%H_%M_%S"), name), 'w')
    print("--------------{0} Sampling Method---------------".format(name))

    oversample = method
    X_over, y_over = oversample.fit_resample(X_train,y_train)

    models = make_voting_classifier(models_dict)
    voting_compare_models_result(models, X_over, y_over, X_val, y_val)
    sys.stdout.close() 