#### basic modeling
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2
############################################################################################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier as GBM
from sklearn.linear_model import RidgeClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from xgboost import XGBRFClassifier
from lightgbm import LGBMClassifier

train_data_user_encoded = pd.read_csv('train_data_user_encoded.csv', index_col='key')
predict_data_user_encoded = pd.read_csv('predict_data_user_encoded.csv', index_col='key')
submission_df = pd.read_csv('C:\py_workspace\SK_AI_CDS_1\Submission\CDS_submission.csv')

train_data_user_encoded_new = pd.read_csv('train_data_user_encoded_mean_stand_new.csv', index_col='key')
predict_data_user_encoded_new = pd.read_csv('predict_data_user_encoded_mean_stand_new.csv', index_col='key')

def training_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    print('f1 score : ', f1_score(y_val, y_pred))
    print('-------------------------------------------------------')
    print('accuracy : ', accuracy_score(y_val, y_pred))
    print('-------------------------------------------------------')
    print(classification_report(y_val, y_pred))
    print('-------------------------------------------------------')
    print('confusion matrix : \n', confusion_matrix(y_val, y_pred))

def check_model_result(models):
    for model in models:
        print('\n')
        print('###### {0} result #########'.format(str(model.__class__.__name__)))
        training_model(model)    
 
X = train_data_user_encoded.drop('Repurchase', axis = 1)
y = train_data_user_encoded['Repurchase']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

X = train_data_user_encoded_new.drop('Repurchase', axis = 1)
y = train_data_user_encoded_new['Repurchase']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

check_model_result(models)

#### oversampling + undersampling
def training_model_over(model):
    model.fit(X_over, y_over)
    y_pred = model.predict(X_val)
    
    print('f1 score : ', f1_score(y_val, y_pred))
    print('-------------------------------------------------------')
    print('accuracy : ', accuracy_score(y_val, y_pred))
    print('-------------------------------------------------------')
    print(classification_report(y_val, y_pred))
    print('-------------------------------------------------------')
    print('confusion matrix : \n', confusion_matrix(y_val, y_pred))
    
    
def check_model_result_over(models):
    for model in models:
        print('\n')
        print('###### {0} result #########'.format(str(model.__class__.__name__)))
        training_model_over(model)
        
X = train_data_user_encoded.drop('Repurchase', axis = 1)
y = train_data_user_encoded['Repurchase']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
oversample = SMOTEENN(n_jobs=-1, random_state=0)

X_over, y_over = oversample.fit_resample(X_train,y_train)

X = train_data_user_encoded_new.drop('Repurchase', axis = 1)
y = train_data_user_encoded_new['Repurchase']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
oversample = SMOTEENN(n_jobs=-1, random_state=0)

X_over, y_over = oversample.fit_resample(X_train,y_train)

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNN()
xgb = XGBRFClassifier()
gbm = GBM()
lgbm = LGBMClassifier()
ridge = RidgeClassifier()
svc = SVC()

models = [dt, rf, knn, xgb, gbm, lgbm, ridge, svc]

check_model_result_over(models)



#### voting classifier

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNN()
xgb = XGBRFClassifier()
gbm = GBM()
lgbm = LGBMClassifier()
ridge = RidgeClassifier()
svc = SVC()

models = [dt, rf, knn, xgb, gbm, lgbm, ridge, svc]

voting_c = VotingClassifier(estimators=[('rf',rf),  ('xgboost', xgboost), ('gbm' , gbm), ('lgbm' , lgbm), ('ridge' , ridge), ('svc' , svc), ('nb' , nb)], n_jobs=7)

check_model_result_over([voting_c])


#### 일반 sampling model + 변형 sampling model

X = train_data_user_encoded_new.drop('Repurchase', axis = 1)
y = train_data_user_encoded_new['Repurchase']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
oversample = SMOTEENN(n_jobs=-1, random_state=0)

X_over, y_over = oversample.fit_resample(X_train,y_train)

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNN()
xgboost = XGBRFClassifier()
gbm = GBM()
lgbm = LGBMClassifier()
ridge = RidgeClassifier()
svc = SVC()
lr = LogisticRegression()
nb = GaussianNB()

voting_1 = VotingClassifier(estimators=[('rf',rf),  ('xgboost', xgboost), ('gbm' , gbm), ('lgbm' , lgbm), ('ridge' , ridge), ('svc' , svc), ('nb' , nb)], n_jobs=7)
voting_2 = VotingClassifier(estimators=[('rf',rf),  ('xgboost', xgboost), ('gbm' , gbm), ('lgbm' , lgbm), ('ridge' , ridge), ('svc' , svc), ('nb' , nb)], n_jobs=7)

voting_1.fit(X_train, y_train)
voting_2.fit(X_over, y_over)
voting_1_pred = voting_1.predict(X_val)
voting_2_pred = voting_2.predict(X_val)

pred_df = pd.concat((pd.Series(voting_1_pred), pd.Series(voting_2_pred)), axis = 1)
pred_df.columns = ['voting_1', 'voting_2']

pred_df = pd.concat((pd.Series(voting_1_pred), pd.Series(voting_2_pred)), axis = 1)
pred_df.columns = ['voting_1', 'voting_2']

pred_df['total'] = np.where((pred_df['voting_1'] == 0) & (pred_df['voting_2'] ==0) , 0, 1)



#### SVM class weight check
test_svm = SVC(class_weight='balanced')
test_svm.fit(X_train,y_train)

test_svm.fit(X_train,y_train)

test_pred = test_svm.predict(X_val)

print('f1 score : ', f1_score(y_val, test_pred))
print('-------------------------------------------------------')
print('accuracy : ', accuracy_score(y_val, test_pred))
print('-------------------------------------------------------')
print(classification_report(y_val, test_pred))
print('-------------------------------------------------------')
print('confusion matrix : \n', confusion_matrix(y_val, test_pred))

#### 결과 제출용
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNN()
xgboost = XGBRFClassifier()
gbm = GBM()
lgbm = LGBMClassifier()
ridge = RidgeClassifier()
svc = SVC()
lr = LogisticRegression()
nb = GaussianNB()

train_data_user_encoded_new = pd.read_csv('train_data_user_encoded_mean_stand.csv', index_col='key')
predict_data_user_encoded_new = pd.read_csv('predict_data_user_encoded_mean_stand.csv', index_col='key')

X_final_train = train_data_user_encoded_new.drop('Repurchase', axis = 1)
y_final_train = train_data_user_encoded_new['Repurchase']

X_final_predict = predict_data_user_encoded_new

from imblearn.combine import SMOTETomek, SMOTEENN
oversample = SMOTEENN(n_jobs=-1)

X_final_train_over, y_final_train_over = oversample.fit_resample(X_final_train,y_final_train)

voting_c = VotingClassifier(estimators=[('lgbm' , lgbm), ('svc' , svc),('nb' , nb)], n_jobs=3)
voting_c.fit(X_final_train_over, y_final_train_over)
pred = voting_c.predict(X_final_predict)

submission_df['CHURN'] = pred