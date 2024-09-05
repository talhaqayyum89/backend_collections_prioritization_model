import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from boruta import BorutaPy
from pycaret.classification import *
from feature_selection import get_recommended_features

print("Libraries successfully imported")


def split_training_test(features, labels, test_size):
    print(f"The shape of feature set is {features.shape}")
    print(f"The shape of labels is {labels.shape}")
    print(f"The final model features are {features.columns}")

    # split date into train & test

    train_X, test_X, train_Y, test_Y = train_test_split(features, labels, test_size=test_size, stratify=labels,
                                                        random_state=42)

    assert (len(train_X) == len(train_Y))

    return train_X, test_X, train_Y, test_Y


def run_training_pycaret(X_train, y_train, X_test, y_test, target, experiment_name):
    """

    Initiates the pycaret training sequence


    """

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    try:
        classifier_ = setup(data=train_data, target=target, experiment_name=experiment_name,
                            test_data=test_data, combine_rare_levels=True)

    except NameError as e:
        print(e)
        print('training job encountered an error')


def create_best_model():
    """
    returns best model object

    """

    select_best = compare_models(n_select=1)

    return select_best


def save_model_artefact(select_best):
    """
    returns a saved_model

    """

    saved_model = save_model(select_best, 'collections_model_v1')

    return saved_model


def get_features():
    recommended = get_recommended_features()
    recommended = recommended.drop(columns=['AccountState'], axis=1, inplace=False)
    return recommended


def modeling(recommended, labels):
    train_X, test_X, train_Y, test_Y = split_training_test(recommended, labels, 0.2)
    run_training_pycaret(train_X, train_Y, test_X, test_Y, 'Flagnew', 'log_1')
    best_model = create_best_model()
    saved_model = save_model_artefact(best_model)
    return best_model
