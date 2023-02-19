import logging
from os import path
import churn_library as cls
import numpy as np

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(dataframe):
    '''
    test perform eda function
    '''
    try:
        cls.perform_eda(dataframe)
    except Exception as err:
        logging.error("Testing perform_eda: FAILED")
        raise err

    try:
        assert path.exists("./images/eda/Churn.png")
        assert path.exists("./images/eda/Customer_Age.png")
        assert path.exists("./images/eda/Heatmap.png")
        assert path.exists("./images/eda/Months_on_book.png")
    except AssertionError as err:
        logging.error("Testing perform_eda: Missing generated eda plots")
        raise err


def test_encoder_helper(dataframe, category_lst, response):
    '''
    test encoder helper
    '''
    try:
        dataframe_encoded = cls.encoder_helper(
            dataframe, category_lst, response)
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: FAILED")
        raise err

    try:
        assert isinstance(
            dataframe_encoded[f'{category_lst[0]}_{response}'][0], np.float64)
    except Exception as err:
        logging.error("Testing encoder_helper: Feature encoded failed")
        raise err


def test_perform_feature_engineering(dataframe, response):
    '''
    test perform_feature_engineering
    '''
    try:
        cls.perform_feature_engineering(dataframe, response)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_feature_engineering: FAILED")
        raise err


def test_train_models(x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        cls.train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: FAILED")
        raise err

    try:
        assert path.exists("./images/results/lr_test_results.png")
        assert path.exists("./images/results/lr_train_results.png")
        assert path.exists("./images/results/rf_test_results.png")
        assert path.exists("./images/results/rf_train_results.png")
    except AssertionError as err:
        logging.error("Testing train_models: Missing classification report")
        raise err

    try:
        assert path.exists("./images/results/Feature_importance_rfc.png")
    except AssertionError as err:
        logging.error("Testing train_models: Missing feature importance image")
        raise err

    try:
        assert path.exists("./images/results/ROC_curves.png")
    except AssertionError as err:
        logging.error("Testing train_models: Missing ROC curves plot image")
        raise err

    try:
        assert path.exists("./models/logistic_model.pkl")
        assert path.exists("./models/rfc_model.pkl")
    except AssertionError as err:
        logging.error(
            "Testing train_models: model is not saved to mdodels folder correctly")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    bank_data_dataframe = cls.import_data("./data/bank_data.csv")
    test_eda(bank_data_dataframe)
    test_perform_feature_engineering(bank_data_dataframe, "Churn")
    x_train_, x_test_, y_train_, y_test_ = cls.perform_feature_engineering(
        bank_data_dataframe, "Churn")

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    test_encoder_helper(bank_data_dataframe, cat_columns, "Churn")

    test_train_models(x_train_, x_test_, y_train_, y_test_)
