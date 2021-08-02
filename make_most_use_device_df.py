import os 
import pandas as pd

base_path = os.path.normpath('C:\py_workspace\SK_AI_CDS_1')
train_path = os.path.join(base_path, 'train')
predict_path = os.path.join(base_path, 'predict')

def make_most_use_device_df(return_df_name):

    if return_df_name == 'train':
        bookmark_df = pd.read_csv(os.path.join(train_path, '{}_bookmark.csv'.format(return_df_name)), 
        dtype = {'hour' : str, 'devicetype' : str}, parse_dates=['dates'], infer_datetime_format=True)

        service_df = pd.read_csv(os.path.join(train_path, '{}_service.csv'.format(return_df_name)), 
        dtype = {'agegroup' : str, 'chargetypeid' : str}, parse_dates = ['registerdate', 'enddate'], infer_datetime_format=True)

    elif return_df_name == 'predict':
        bookmark_df = pd.read_csv(os.path.join(predict_path, '{}_bookmark.csv'.format(return_df_name)), 
        dtype = {'hour' : str, 'devicetype' : str}, parse_dates=['dates'], infer_datetime_format=True)

        service_df = pd.read_csv(os.path.join(predict_path, '{}_service.csv'.format(return_df_name)), 
        dtype = {'agegroup' : str, 'chargetypeid' : str}, parse_dates = ['registerdate', 'enddate'], infer_datetime_format=True)
    

    temp1 = bookmark_df[['uno', 'programid']]
    temp1 = temp1[~temp1.duplicated()]
    bookmark_unique_uno_program_df = temp1.groupby(by = 'uno').count()

    temp2 = bookmark_df[['uno', 'dates']]
    temp2 = temp2[~temp2.duplicated()]
    bookmark_unique_uno_dates_df = temp2.groupby(by = 'uno').count()

    temp3 = bookmark_df[['uno', 'viewtime']]
    bookmark_unique_uno_viewtime_df = temp3.groupby(by = 'uno').sum()

    temp4 = bookmark_df[['uno', 'devicetype']]
    temp4 = temp4[~temp4.duplicated()]
    bookmark_unique_uno_device_num_df = temp4.groupby(by = 'uno').count()

    test = bookmark_df[['uno', 'viewtime', 'devicetype']]
    bookmark_unique_uno_test_df = test.groupby(by = ['uno', 'devicetype']).sum()
    test2 = bookmark_unique_uno_test_df
    test3 = test2.reset_index()
    uno_lists = list(test3['uno'].unique())

    devicetype_dict = {}
    for uno in uno_lists:
        uno_data = test3[test3['uno'] == uno]
        max_viewtime = uno_data['viewtime'].max()
        devicetype = uno_data[uno_data['viewtime'] == max_viewtime]['devicetype']
    #     devicetype.index = [uno]
        devicetype_dict[uno] = devicetype.values[0]
        
    bookmark_unique_uno_most_use_device_df = pd.DataFrame(devicetype_dict.values(), index = devicetype_dict.keys(), columns = ['most_use_devicetype'])
    bookmark_unique_uno_most_use_device_df.index.name = 'uno'
    bookmark_unique_uno_most_use_device_df.to_csv('{}_bookmark_unique_uno_most_use_device_df.csv'.format(return_df_name))

make_most_use_device_df('train')