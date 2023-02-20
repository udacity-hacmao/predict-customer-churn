'''
This is the Python module contain function to identify credit card customers that are most likely to churn.

Author: HiepNT25@fsoft.com.vn
Date: 20/02/2023
'''
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, train_test_split

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)

    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    print(dataframe.head())
    print(dataframe.shape)
    print(dataframe.isnull().sum())

    eda_path = './images/eda'

    def plt_hist(feature):
        plt.figure(figsize=(20, 10))
        dataframe[feature].hist()
        plt.savefig(f'{eda_path}/{feature}.png')

    def plt_heatmap():
        plt.figure(figsize=(20, 10))
        sns.heatmap(dataframe.corr(numeric_only=True),
                    annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(f'{eda_path}/Heatmap.png')

    plt_hist('Churn')
    plt_hist('Customer_Age')
    plt_hist('Months_on_book')
    plt_heatmap()


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe`=
            category_lst: list of columns that contain categorical features
            response: string of response name 
                      [optional argument that could be used for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    '''
    for feature in category_lst:
        feature_lst = []
        feature_groups = dataframe.groupby(feature).mean()[response]

        for val in dataframe[feature]:
            feature_lst.append(feature_groups.loc[val])

        dataframe[f'{feature}_{response}'] = feature_lst
    return dataframe


def perform_feature_engineering(dataframe, response):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name 
                [optional argument that could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # encode categorical columes
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    dataframe_encoded = encoder_helper(dataframe, cat_columns, response)

    # generate x_data and y_data
    y_data = dataframe[response]
    x_data = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    x_data[keep_cols] = dataframe_encoded[keep_cols]

    # Split data into training data and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores
    def export_classification_report(y_true, y_pred, output_path):
        '''
        Export classification report to image image
        '''
        plt.figure()
        clf_report = classification_report(y_true, y_pred, output_dict=True)
        fig = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :], annot=True)
        fig.figure.savefig(output_path)

    output_dir = "./images/results"
    print('random forest results')
    print('test results')
    export_classification_report(
        y_test, y_test_preds_rf, f'{output_dir}/rf_test_results.png')
    print('train results')
    export_classification_report(
        y_train, y_train_preds_rf, f'{output_dir}/rf_train_results.png')

    print('logistic regression results')
    print('test results')
    export_classification_report(
        y_test, y_test_preds_lr, f'{output_dir}/lr_test_results.png')
    print('train results')
    export_classification_report(
        y_train, y_train_preds_lr, f'{output_dir}/lr_train_results.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")

    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Plot Roc curve
    plt.figure(figsize=(20,5))
    ax_ = plt.gca()
    RocCurveDisplay.from_estimator(lrc, x_test, y_test, ax=ax_, alpha=0.8)
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, x_test, y_test, ax=ax_, alpha=0.8)
    plt.savefig("./images/results/ROC_curves.png")

    # Plot classification report
    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

    # Plot fefature importance for cv_rfc
    feature_importance_plot(cv_rfc.best_estimator_, x_train,
                            "./images/results/Feature_importance_rfc.png")

    # Save models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':
    bank_data_dataframe = import_data('./data/bank_data.csv')
    perform_eda(bank_data_dataframe)
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(
        bank_data_dataframe, "Churn")
    train_models(x_train_, x_test_, y_train_, y_test_)
