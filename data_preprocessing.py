import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from datetime import datetime, timedelta

base_path = os.path.normpath('')
train_path = os.path.join(base_path, 'train')
predict_path = os.path.join(base_path, 'predict')
code_path = os.path.join(base_path, 'code')

train_bookmark_df = pd.read_csv(os.path.join(train_path, 'train_bookmark.csv'), dtype = {'hour' : str, 'devicetype' : str}, parse_dates=['dates'], infer_datetime_format=True)
train_service_df = pd.read_csv(os.path.join(train_path, 'train_service.csv'), dtype = {'agegroup' : str, 'chargetypeid' : str}, parse_dates = ['registerdate', 'enddate'], infer_datetime_format=True)
coin_df = pd.read_csv(os.path.join(code_path, 'coin.csv'), dtype = {'paymenttypeid' : str}, parse_dates=['registerdate', 'coinexpiredate'], infer_datetime_format=True)
content_info_df = pd.read_csv(os.path.join(code_path, 'content_info.csv'))
movie_info_df = pd.read_csv(os.path.join(code_path, 'movie_info.csv'))
product_code_df = pd.read_excel(os.path.join(base_path, 'Column\column_code_info.xlsx'), sheet_name = 'Productcode')


#### data load
def data_overview(df):
    print(df.head())
    print("------------------------------------------------------")
    print(df.info())
    print("------------------------------------------------------")
    print(df.shape)
    print("------------------------------------------------------")
    
data_overview(train_bookmark_df)
data_overview(train_service_df)
data_overview(coin_df)
data_overview(content_info_df)
data_overview(movie_info_df)



#### check null value
def check_null(df):
    return_df = pd.DataFrame({'null value' : df.isnull().sum(), 'null ratio' : df.isnull().sum() / df.shape[0]})
    return_df = return_df.sort_values(by = 'null ratio', ascending = False)
    return return_df

print("unique train_bookmark 사용자 수 : ", len(train_bookmark_df['uno'].unique()))
train_bookmark_null_check_df = check_null(train_bookmark_df)
print(train_bookmark_null_check_df)

print('unique train_service 사용자 수 : ', len(train_service_df['uno'].unique()))
train_service_null_check_df = check_null(train_service_df)
print(train_service_null_check_df)

coin_null_df = check_null(coin_df)
print(coin_null_df)

content_null_df = check_null(content_info_df)
print(content_null_df)

movie_null_df = check_null(movie_info_df)
print(movie_null_df)

#### Data Merge
#### bookmark의 모든 사용자는  service의 모든 사용자에 포함
print('train_bookmark unique 사용자에 포함되지 않는 train_service unique 사용자 수 : ', sum(~pd.Series(train_service_df['uno'].unique()).isin(train_bookmark_df['uno'].unique())))
print('train_service unique 사용자에 포함되지 않는 train_bookmark unique 사용자 수 : ', sum(~pd.Series(train_bookmark_df['uno'].unique()).isin(train_service_df['uno'].unique())))

print('개인별(uno 별) train_bookmark_df 중복 개수')
train_bookmark_df.groupby('uno').count()

check_train_bookmark_duplicated_df = train_bookmark_df.groupby('uno').count()
check_train_bookmark_duplicated_df.describe()

test = train_bookmark_df[['uno', 'programid']]
test = test[~test.duplicated()]
train_bookmark_unique_uno_program_df = test.groupby(by = 'uno').count()
print('사용자 별(uno 별) 시청한 program id 개수')
train_bookmark_unique_uno_program_df

test = train_bookmark_df[['uno', 'dates']]
test = test[~test.duplicated()]
train_bookmark_unique_uno_dates_df = test.groupby(by = 'uno').count()
print('사용자 별(uno 별) 시청한 dates 횟수')
train_bookmark_unique_uno_dates_df

test = train_bookmark_df[['uno', 'viewtime']]
train_bookmark_unique_uno_viewtime_df = test.groupby(by = 'uno').sum()
print('사용자 별(uno 별) 시청한 시청시간 총합')
train_bookmark_unique_uno_viewtime_df

test = train_bookmark_df[['uno', 'devicetype']]
test = test[~test.duplicated()]
train_bookmark_unique_uno_device_num_df = test.groupby(by = 'uno').count()
print('사용자 별(uno 별) 사용한 시청 기기의 개수')
train_bookmark_unique_uno_device_num_df

test = train_bookmark_df[['uno', 'channeltype']]
test = test[~test.duplicated()]
train_bookmark_unique_uno_channeltype_num_df = test.groupby(by = 'uno').count()
print('사용자 별(uno 별) 사용한 채널 타입의 개수')
train_bookmark_unique_uno_channeltype_num_df

#### 개인별 viewtime이 가장 큰 devicetype - csv파일 만드는 python 파일 존재
test = train_bookmark_df[['uno', 'viewtime', 'devicetype']]
train_bookmark_unique_uno_test_df = test.groupby(by = ['uno', 'devicetype']).sum()
print('사용자 별(uno 별) 가장 많이 사용한 시청 기기')
test2 = train_bookmark_unique_uno_test_df
train_bookmark_unique_uno_most_use_device_df = pd.read_csv('train_bookmark_unique_uno_most_use_device_df.csv', index_col=0)
train_bookmark_unique_uno_most_use_device_df.index.name = 'uno'
train_bookmark_unique_uno_most_use_device_df

print('개인별(uno 별) train_service_df 중복 개수')
train_service_df.groupby('uno').count()

check_train_service_duplicated_df = train_service_df.groupby('uno').count()
print('uno별 train_servce_df 값이 2 이상인 데이터')
# print(check_train_service_duplicated_df[check_train_service_duplicated_df.iloc[:, 0] > 1])
sns.countplot(check_train_service_duplicated_df.iloc[:,0][check_train_service_duplicated_df.iloc[:,0] > 1])
check_train_service_duplicated_df[check_train_service_duplicated_df.iloc[:, 0] > 1].describe()

#### 이용권을 등록기간 이후 3주 이상 유지한 고객 목록
train_service_df_over3w = train_service_df[train_service_df['registerdate'] + timedelta(weeks = 3) - timedelta(days = 1) < train_service_df['enddate']]

def product_class(desc):
    return desc.split()[0]

product_code_df['product_class'] = product_code_df['Productcode_name'].map(lambda x : product_class(x))
product_code_df = product_code_df.drop(['description', 'Productcode_name'], axis =1)

#### 사용자별 총 시청 시간, 시청 날짜 횟수, 시청 프로그램 개수, 재가입 횟수 변수 추가 
train_bookmark_service_merge =  pd.merge(left = train_service_df_over3w, right = train_bookmark_unique_uno_viewtime_df, on = 'uno', how = 'outer')
train_bookmark_service_merge =  pd.merge(left = train_bookmark_service_merge, right = train_bookmark_unique_uno_dates_df, on = 'uno', how = 'outer')
train_bookmark_service_merge =  pd.merge(left = train_bookmark_service_merge, right = train_bookmark_unique_uno_program_df, on = 'uno', how = 'outer')
train_bookmark_service_merge =  pd.merge(left = train_bookmark_service_merge, right = train_bookmark_unique_uno_device_num_df, on = 'uno', how = 'outer')
train_bookmark_service_merge =  pd.merge(left = train_bookmark_service_merge, right = train_bookmark_unique_uno_channeltype_num_df, on = 'uno', how = 'outer')
train_bookmark_service_merge =  pd.merge(left = train_bookmark_service_merge, right = train_bookmark_unique_uno_most_use_device_df, on = 'uno', how = 'outer')
train_bookmark_service_merge = pd.merge(left = train_bookmark_service_merge, right = check_train_service_duplicated_df['registerdate'], on = 'uno', how = 'outer')
train_bookmark_service_merge = pd.merge(left = train_bookmark_service_merge, right= product_code_df, left_on='productcode', right_on='Productcode', how = 'left')
train_bookmark_service_merge.rename(columns = {'registerdate_x' : 'registerdate', 'registerdate_y' : 'rejoin_num',
                                               'channeltype' : 'using_channeltype_num', 'devicetype' : 'using_device_num',
                                               'dates' : 'watching_dates_num',
                                              'programid' : 'watching_progrm_num', 'viewtime' : 'total_watching_time'}, inplace=True)


train_bookmark_service_merge
train_bookmark_service_merge.rename(columns = {}, inplace=True)

train_bookmark_service_merge.to_csv('train_bookmark_service_merge_new.csv', index = False)

train_data = train_bookmark_service_merge.copy()
check_null(train_data)

def age_group(x):
    under_20 = ['0','5','10','15']
    age_20 = ['20', '25']
    age_30 = ['30', '35']
    age_40 = ['40', '45']
    age_50 = ['50', '55']
    
    if x in under_20:
        return 'under_20'
    elif x in age_20:
        return 'age_20'
    elif x in age_30:
        return 'age_30'
    elif x in age_40:
        return 'age_40'
    elif x in age_50:
        return 'age_50'
    else:
        return 'over_50'
    
train_data['age_class'] = train_data['agegroup'].map(lambda x : age_group(x))
train_data.drop('agegroup',axis = 1, inplace=True)

#### 100 미만은 달러 값이므로 이를 원화 결과로 변경 - 21/07/08 기준 환율 1$ ~ 1142
train_data['pgamount'][train_data['pgamount'] < 100] = train_data['pgamount'][train_data['pgamount'] < 100] * 1142

#### 필요없는 column 제거
drop_col_lists = ['promo_100', 'enddate', 'Productcode']
train_data.drop(drop_col_lists, axis = 1, inplace=True)

train_data['most_use_devicetype'] = train_data['most_use_devicetype'].astype('str')
train_data.info()

# #### null값 처리 - 평균
train_data['gender'] = train_data['gender'].fillna('N')
train_data['coinReceived'] = train_data['coinReceived'].fillna('X')
train_data['isauth'] = train_data['isauth'].fillna('N')
# train_data['section'] = train_data['section'].fillna('N')
# train_data['programid'] = train_data['programid'].fillna('N')
train_data['using_channeltype_num'] = train_data['using_channeltype_num'].fillna(train_data['using_channeltype_num'].mean())
train_data['using_device_num'] = train_data['using_device_num'].fillna(train_data['using_device_num'].mean())
train_data['watching_progrm_num'] = train_data['watching_progrm_num'].fillna(train_data['watching_progrm_num'].mean())
train_data['watching_dates_num'] = train_data['watching_dates_num'].fillna(train_data['watching_dates_num'].mean())
train_data['total_watching_time'] = train_data['total_watching_time'].fillna(train_data['total_watching_time'].mean())
train_data['most_use_devicetype'] = train_data['most_use_devicetype'].fillna(train_data['most_use_devicetype'].value_counts().idxmax())

# # #### null값 처리 - 중앙값
# train_data['gender'] = train_data['gender'].fillna('N')
# train_data['coinReceived'] = train_data['coinReceived'].fillna('X')
# train_data['isauth'] = train_data['isauth'].fillna('N')
# # train_data['section'] = train_data['section'].fillna('N')
# # train_data['programid'] = train_data['programid'].fillna('N')
# train_data['using_channeltype_num'] = train_data['using_channeltype_num'].fillna(train_data['using_channeltype_num'].median())
# train_data['using_device_num'] = train_data['using_device_num'].fillna(train_data['using_device_num'].median())
# train_data['watching_progrm_num'] = train_data['watching_progrm_num'].fillna(train_data['watching_progrm_num'].median())
# train_data['watching_dates_num'] = train_data['watching_dates_num'].fillna(train_data['watching_dates_num'].median())
# train_data['total_watching_time'] = train_data['total_watching_time'].fillna(train_data['total_watching_time'].median())
# train_data['most_use_devicetype'] = train_data['most_use_devicetype'].fillna(train_data['most_use_devicetype'].value_counts().idxmax())


#### Encode Categorical data to Numerical data & data preprocess
train_data.to_csv('train_data_new.csv', index = False)

from sklearn.preprocessing import OneHotEncoder


def to_one_hot_train(df):
    one_hot_df = pd.DataFrame()
    col_encoder = {}
    col_lists = list(df.dtypes[df.dtypes == 'category'].index)
    
    for col in col_lists:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(np.array(df[col]).reshape(-1,1))
        col_encoder[col] = encoder.fit(np.array(df[col]).reshape(-1,1))
        add_df = pd.DataFrame(encoder.transform(np.array(df[col]).reshape(-1,1)).toarray())
        add_df = add_df.add_prefix(str(col) + '_')
        one_hot_df = pd.concat([one_hot_df, add_df], axis = 1)
    
    return one_hot_df, col_encoder

train_data_user = train_data.drop(['uno', 'registerdate', 'productcode', 'Repurchase'], axis = 1) #### 식별키이므로 제거
train_data_user[train_data_user.dtypes[train_data_user.dtypes == 'object'].index] = train_data_user[train_data_user.dtypes[train_data_user.dtypes == 'object'].index].astype('category')
one_hot_df_train, encoder_dict = to_one_hot_train(train_data_user)

train_data_user_encoded = train_data_user[train_data_user.dtypes[train_data_user.dtypes != 'category'].index]
train_data_user_encoded = pd.concat([train_data_user_encoded, one_hot_df_train], axis = 1)
train_data_user_encoded['Repurchase'] = np.where(train_data['Repurchase'] == 'X', 1, 0)    #### 학습을 위해 Repurchase 수정
train_data_user_encoded['key'] = train_data['uno']+'|'+train_data['registerdate'].astype('str')+'|'+train_data['productcode']  #### 식별키 다시 추가
train_data_user_encoded = train_data_user_encoded.set_index('key')
train_data_user_encoded

from sklearn.preprocessing import StandardScaler, MinMaxScaler

numeric_col_lists = list(train_data_user_encoded.columns[:8])

for col in numeric_col_lists:
    st_encoder = StandardScaler()
    transformed_return = st_encoder.fit_transform(np.array(train_data_user_encoded[col]).reshape(-1,1))
    train_data_user_encoded[col] = transformed_return
    
train_data_user_encoded.to_csv('train_data_user_encoded_mean_stand_new.csv')


#### train preprocess predict data 적용
predict_bookmark_df = pd.read_csv(os.path.join(predict_path, 'predict_bookmark.csv'), dtype = {'hour' : str, 'devicetype' : str}, parse_dates=['dates'], infer_datetime_format=True)
predict_service_df = pd.read_csv(os.path.join(predict_path, 'predict_service.csv'), dtype = {'agegroup' : str, 'chargetypeid' : str}, parse_dates = ['registerdate', 'enddate'], infer_datetime_format=True)

test = predict_bookmark_df[['uno', 'programid']]
test = test[~test.duplicated()]
predict_bookmark_unique_uno_program_df = test.groupby(by = 'uno').count()

test = predict_bookmark_df[['uno', 'dates']]
test = test[~test.duplicated()]
predict_bookmark_unique_uno_dates_df = test.groupby(by = 'uno').count()

test = predict_bookmark_df[['uno', 'viewtime']]
predict_bookmark_unique_uno_viewtime_df = test.groupby(by = 'uno').sum()

test = predict_bookmark_df[['uno', 'devicetype']]
test = test[~test.duplicated()]
predict_bookmark_unique_uno_device_num_df = test.groupby(by = 'uno').count()

test = predict_bookmark_df[['uno', 'channeltype']]
test = test[~test.duplicated()]
predict_bookmark_unique_uno_channeltype_num_df = test.groupby(by = 'uno').count()

predict_bookmark_unique_uno_most_use_device_df = pd.read_csv('predict_bookmark_unique_uno_most_use_device_df.csv', index_col='uno')

check_predict_service_duplicated_df = predict_service_df.groupby('uno').count()

predict_service_df_over3w = predict_service_df[predict_service_df['registerdate'] + timedelta(weeks = 3) - timedelta(days = 1) < predict_service_df['enddate']]

predict_bookmark_service_merge =  pd.merge(left = predict_service_df_over3w, right = predict_bookmark_unique_uno_viewtime_df, on = 'uno', how = 'outer')
predict_bookmark_service_merge =  pd.merge(left = predict_bookmark_service_merge, right = predict_bookmark_unique_uno_dates_df, on = 'uno', how = 'outer')
predict_bookmark_service_merge =  pd.merge(left = predict_bookmark_service_merge, right = predict_bookmark_unique_uno_program_df, on = 'uno', how = 'outer')
predict_bookmark_service_merge =  pd.merge(left = predict_bookmark_service_merge, right = predict_bookmark_unique_uno_device_num_df, on = 'uno', how = 'outer')
predict_bookmark_service_merge =  pd.merge(left = predict_bookmark_service_merge, right = predict_bookmark_unique_uno_channeltype_num_df, on = 'uno', how = 'outer')
predict_bookmark_service_merge =  pd.merge(left = predict_bookmark_service_merge, right = predict_bookmark_unique_uno_most_use_device_df, on = 'uno', how = 'outer')
predict_bookmark_service_merge = pd.merge(left = predict_bookmark_service_merge, right = check_predict_service_duplicated_df['registerdate'], on = 'uno', how = 'outer')
predict_bookmark_service_merge = pd.merge(left = predict_bookmark_service_merge, right = product_code_df, left_on='productcode', right_on='Productcode', how = 'left')
predict_bookmark_service_merge.rename(columns = {'registerdate_x' : 'registerdate','registerdate_y' : 'rejoin_num','channeltype' : 'using_channeltype_num', 'devicetype' : 'using_device_num',
                                               'dates' : 'watching_dates_num',
                                              'programid' : 'watching_progrm_num', 'viewtime' : 'total_watching_time'}, inplace=True)


predict_bookmark_service_merge.to_csv('predict_bookmark_service_merge_new.csv', index = False)

check_null(predict_bookmark_service_merge)

predict_data = predict_bookmark_service_merge.copy()

# #### null값 처리 - 평균
predict_data['gender'] = predict_data['gender'].fillna('N')
predict_data['coinReceived'] = predict_data['coinReceived'].fillna('X')
predict_data['isauth'] = predict_data['isauth'].fillna('N')
# train_data['section'] = train_data['section'].fillna('N')
# train_data['programid'] = train_data['programid'].fillna('N')
predict_data['using_channeltype_num'] = predict_data['using_channeltype_num'].fillna(predict_data['using_channeltype_num'].mean())
predict_data['using_device_num'] = predict_data['using_device_num'].fillna(predict_data['using_device_num'].mean())
predict_data['watching_progrm_num'] = predict_data['watching_progrm_num'].fillna(predict_data['watching_progrm_num'].mean())
predict_data['watching_dates_num'] = predict_data['watching_dates_num'].fillna(predict_data['watching_dates_num'].mean())
predict_data['total_watching_time'] = predict_data['total_watching_time'].fillna(predict_data['total_watching_time'].mean())
predict_data['most_use_devicetype'] = predict_data['most_use_devicetype'].fillna(predict_data['most_use_devicetype'].value_counts().idxmax())

# # #### null값 처리 - 중앙값
# predict_data['gender'] = predict_data['gender'].fillna('N')
# predict_data['coinReceived'] = predict_data['coinReceived'].fillna('X')
# predict_data['isauth'] = predict_data['isauth'].fillna('N')
# # train_data['section'] = train_data['section'].fillna('N')
# # train_data['programid'] = train_data['programid'].fillna('N')
# predict_data['using_channeltype_num'] = predict_data['using_channeltype_num'].fillna(predict_data['using_channeltype_num'].median())
# predict_data['using_device_num'] = predict_data['using_device_num'].fillna(predict_data['using_device_num'].median())
# predict_data['watching_progrm_num'] = predict_data['watching_progrm_num'].fillna(predict_data['watching_progrm_num'].median())
# predict_data['watching_dates_num'] = predict_data['watching_dates_num'].fillna(predict_data['watching_dates_num'].median())
# predict_data['total_watching_time'] = predict_data['total_watching_time'].fillna(predict_data['total_watching_time'].median())
# predict_data['most_use_devicetype'] = predict_data['most_use_devicetype'].fillna(predict_data['most_use_devicetype'].value_counts().idxmax())

def age_group(x):
    under_20 = ['0','5','10','15']
    age_20 = ['20', '25']
    age_30 = ['30', '35']
    age_40 = ['40', '45']
    age_50 = ['50', '55']
    
    if x in under_20:
        return 'under_20'
    elif x in age_20:
        return 'age_20'
    elif x in age_30:
        return 'age_30'
    elif x in age_40:
        return 'age_40'
    elif x in age_50:
        return 'age_50'
    else:
        return 'over_50'
    
predict_data['age_class'] = predict_data['agegroup'].map(lambda x : age_group(x))
predict_data.drop('agegroup',axis = 1, inplace=True)

# age_anomaly_list = ['120', '950']
# predict_data['agegroup'][predict_data['agegroup'].isin(age_anomaly_list)] = '100'
#### 100 미만은 달러 값이므로 이를 원화 결과로 변경 - 21/07/08 기준 환율 1$ ~ 1142
predict_data['pgamount'][predict_data['pgamount'] < 100] = predict_data['pgamount'][predict_data['pgamount'] < 100] * 1142

drop_col_lists = ['promo_100', 'enddate', 'Productcode']
predict_data.drop(drop_col_lists, axis = 1, inplace=True)

predict_data['most_use_devicetype'] = predict_data['most_use_devicetype'].astype('str')
predict_data.to_csv('predict_data_new.csv', index = False)

from sklearn.preprocessing import OneHotEncoder


def to_one_hot_predict(df):
    one_hot_df = pd.DataFrame()
    col_lists = list(df.dtypes[df.dtypes == 'category'].index)
    
    for col in col_lists:
        encoder = encoder_dict[col]
        add_df = pd.DataFrame(encoder.transform(np.array(df[col]).reshape(-1,1)).toarray())
        add_df = add_df.add_prefix(str(col) + '_')
        one_hot_df = pd.concat([one_hot_df, add_df], axis = 1)
    
    return one_hot_df

predict_data_user = predict_data.drop(['uno', 'registerdate', 'productcode', 'Repurchase'], axis = 1) #### 식별키이므로 제거
predict_data_user[predict_data_user.dtypes[predict_data_user.dtypes == 'object'].index] = predict_data_user[predict_data_user.dtypes[predict_data_user.dtypes == 'object'].index].astype('category')
one_hot_df_predict = to_one_hot_predict(predict_data_user)

predict_data_user_encoded = predict_data_user[predict_data_user.dtypes[predict_data_user.dtypes != 'category'].index]
predict_data_user_encoded = pd.concat([predict_data_user_encoded, one_hot_df_predict], axis = 1)
predict_data_user_encoded['key'] = predict_data['uno'] +'|'+predict_data['registerdate'].astype('str')+'|'+predict_data['productcode']   #### 식별키 다시 추가
predict_data_user_encoded = predict_data_user_encoded.set_index('key')
predict_data_user_encoded

### train data와 동일한 scaler encoder 사용
from sklearn.preprocessing import StandardScaler, MinMaxScaler

numeric_col_lists = list(predict_data_user_encoded.columns[:9])

for col in numeric_col_lists:
    transformed_return = st_encoder.transform(np.array(predict_data_user_encoded[col]).reshape(-1,1))
    predict_data_user_encoded[col] = transformed_return
    
predict_data_user_encoded.to_csv('predict_data_user_encoded_mean_stand_new.csv')
