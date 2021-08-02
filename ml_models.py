from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier as GBM
from sklearn.linear_model import RidgeClassifier 
from sklearn.svm import SVC
from xgboost import XGBRFClassifier
from lightgbm import LGBMClassifier
################################################################################################################################################

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNN()
xgboost = XGBRFClassifier()
gbm = GBM()
lgbm = LGBMClassifier()
ridge = RidgeClassifier()
svc = SVC()
lr = LogisticRegression(max_iter=500)
nb = GaussianNB()

def models_dict_make():
    return {'dt' : dt, 'rf' : rf, 'knn' : knn, 'xgboost' : xgboost, 'gbm' : gbm, 'lgbm' : lgbm, 'ridge' : ridge, 'svc' : svc, 'nb' : nb, 'lr' : lr}
    # return {'dt' : dt, 'rf' : rf, 'knn' : knn}