from itertools import combinations
##############################################################
from sklearn.ensemble import VotingClassifier

#### 주어진 모델 목록에서 가질 수 있는 모든 모델 조합들의 이름 list
def model_combination_name(models_dict):
    
    model_combination_name_list = []
    for r in range(len(models_dict)):
        for i in combinations(models_dict, (r+1)):
            model_combination_name_list.append(list(i))

    return model_combination_name_list


#### votingclassifier에 들어가기 위해서는 estimator 변수에 estimator = [('이름' , 분류기)]와 같은 모양을 지녀야 함
#### 모든 모델 조합의 이름을 바탕으로 estimator에 맞는 모양을 만들어 주는 함수
def model_combination(models_dict):
    
    model_combination_name_list = model_combination_name(models_dict)
    model_combination_result = list()

    for k in range(len(model_combination_name_list)):
        temp = {name : models_dict[name] for name in model_combination_name_list[k]}
        model_combination_result.append(list(temp.items()))

    return model_combination_result


#### 모든 조합을 voting classifier에 넣어서 사용 / 1개만 넣더라도 해당 1개짜리 모델을 사용하는 것과 동일한 결과
def make_voting_classifier(models_dict):

    estimator_list = model_combination(models_dict)
    voting_c_list = []
    for i in range(len(estimator_list)):
        estimators_ = estimator_list[i]
        voting_c_list.append(VotingClassifier(estimators=estimators_, n_jobs=len(estimators_)))

    return voting_c_list

### voting에서 estimator 이름만 추출
# str([name[0] for name in voting_c.estimators])