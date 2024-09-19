from data_fetching_func import get_redshift_dataframe
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import category_encoders as ce
from boruta import BorutaPy
import warnings
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

pd.set_option('max_rows', None)
pd.set_option('max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.style.use('ggplot')
%matplotlib inline
color = sns.color_palette()
warnings.filterwarnings("ignore", category=FutureWarning)
print("Libraries successfully imported")


def boruta_select_features(data, target):
    """
    data - {DataFrame}
    target - {Series)

    function that select passes needed features

    into the boruta algorithm

    """

    X = data[[feature for feature in data.columns if feature not in [target]]]
    y = data[target]

    y = y.map({'LazyPayer': 0, 'Soft': 1, 'Hard': 2})
    X_cat = X.select_dtypes(exclude=['int64', 'float64'])
    X_int = X.select_dtypes(include=['int64', 'float64'])

    # categorical encoding transformed applied

    ce_target = ce.TargetEncoder(cols=[i for i in X_cat.columns])
    ce_target.fit(X_cat, y)
    X_transform = ce_target.transform(X_cat, y)

    X_final = pd.concat([X_transform, X_int], axis=1)

    train = X_final.values
    target = y.values.ravel()

    #     return X_final, train, target

    return X_final, y, train, target


def run_boruta_feature_selctor(train, target, final_train_variables):
    """
    train - {DataFrame}
    test - {DataFrame}

    This function runs the boruta feature_selector

    and returns selected features

    """

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42, max_iter=50, perc=90)

    # run the iteration
    boruta_feature_selector.fit(train, target)

    # store the feature importanc e

    green_area = final_train_variables.columns[boruta_feature_selector.support_].to_list()
    blue_area = final_train_variables.columns[boruta_feature_selector.support_weak_].to_list()

    # store recommended variables

    recommended_ = final_train_variables[green_area]

    feature_names = [i for i in final_train_variables.columns]
    boruta_ranking = boruta_feature_selector.ranking_
    selected_features = np.array(feature_names)[boruta_ranking <= 2]

    len_features = len(selected_features)

    boruta_ranking = pd.DataFrame(data=boruta_ranking, index=final_train_variables.columns.values, columns=['values'])
    boruta_ranking['Variable'] = boruta_ranking.index
    boruta_ranking.sort_values(['values'], ascending=True, inplace=True)

    return len_features, boruta_ranking, recommended_


def get_recommended_features():
    data = get_redshift_dataframe()
    features, labels, boruta_train, boruta_test = boruta_select_features(data, 'Flagnew')
    len_features, boruta_ranking, recommended = run_boruta_feature_selctor(boruta_train, boruta_test, features)
    return recommended, labels
