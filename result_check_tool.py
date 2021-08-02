from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
#############################################################################################################################

def get_over_model_result(models, X_over, y_over, X_val, y_val):

    result_dict = dict()
    f1_result = dict()
    sorted_model_list = list()

    for model in models:
        model.fit(X_over, y_over)
        y_pred = model.predict(X_val)
        result_dict[model] = {}
        result_dict[model]['f1_score'] = f1_score(y_val, y_pred)
        f1_result[f1_score(y_val, y_pred)] = model
        result_dict[model]['accuracy_score'] = accuracy_score(y_val, y_pred)
        result_dict[model]['classification_report'] = classification_report(y_val, y_pred)
        result_dict[model]['confusion_matrix'] = confusion_matrix(y_val, y_pred)

    for i in range(len(f1_result)):
        sorted_model_list.append(sorted(f1_result.items(), reverse=True)[i][1])

    return result_dict, sorted_model_list    

def print_model_result(result_dict, model):

    print('f1 score : ', result_dict[model]['f1_score'])
    print('-------------------------------------------------------')
    print('accuracy : ', result_dict[model]['accuracy_score'])
    print('-------------------------------------------------------')
    print(result_dict[model]['classification_report'])
    print('-------------------------------------------------------')
    print('confusion matrix : \n', result_dict[model]['confusion_matrix'])


def compare_models_result(models, X_over, y_over, X_val, y_val):

    result_dict = get_over_model_result(models, X_over, y_over, X_val, y_val)[0]
    sorted_model_list = get_over_model_result(models, X_over, y_over, X_val, y_val)[1]

    for model in sorted_model_list:
        print('\n')
        print('###### {0} result #########'.format(str(model.__class__.__name__)))
        print_model_result(result_dict, model)

def voting_compare_models_result(models, X_over, y_over, X_val, y_val):

    result_dict = get_over_model_result(models, X_over, y_over, X_val, y_val)[0]
    sorted_model_list = get_over_model_result(models, X_over, y_over, X_val, y_val)[1]

    for model in sorted_model_list:
        print('\n')
        print('###### {0} / {1} result #########'.format(str(model.__class__.__name__), str([name[0] for name in model.estimators])))
        print_model_result(result_dict, model)