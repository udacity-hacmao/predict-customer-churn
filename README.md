# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project will identify credit card customers that are most likely to churn.

## Files and data description
```
.
├── Guide.ipynb          # Given: Getting started and troubleshooting tips
├── churn_notebook.ipynb # Given: Contains the code to be refactored
├── churn_library.py     # Main files contains all importance functions
├── churn_script_logging_and_tests.py # Tests and logs
├── README.md            # Project information
├── data                 # Read this data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs                 # Store logs
└── models               # Store models
```

## Running Files
Install library:

```
python3.9 -m pip install -r requirements_py3.9.txt
```

Running model trainning:

```
python3.9 churn_library.py
```

Check results in folder `images` and stored models in folder `models`.

Running test and read log stored in `logs/churn_library.log`

```
python3.9 churn_script_logging_and_tests.py
```



