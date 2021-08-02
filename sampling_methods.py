from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
# ############################################################################################################################


smoteenn = SMOTEENN(n_jobs=-1, random_state=0)
smote = SMOTE(n_jobs=-1, random_state=0)
adasyn = ADASYN(n_jobs=-1, random_state=0)
smotetomek = SMOTETomek(n_jobs=-1, random_state=0)

def sampling_dict_make():
    # return {'SMOTEENN' : smoteenn, 'SMOTE' : smote, 'ADASYN' : adasyn, 'SMOTETomek' : smotetomek}
    return {'SMOTEENN' : smoteenn}