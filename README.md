# Sales forecast for 20 stores for 31 days' period.


Evaluation Metric - [Scaled Mean Absolute Error (sMAE)](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error).


## Content

1. [EDA: omissions analysis & time series clustering](https://github.com/YaninaK/sales-forecast/blob/main/notebooks/01_EDA_omissions_clusters.ipynb)
    * [EDA utilities](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/data/EDA_utilities.py)
    * [Johnson SU transformation](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/data/johnson_su_transformation.py) 
    * [Impute data](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/data/impute_data.py)
    * [Clean outliers](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/data/clean_data.py) 
    * [Time series clusters](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/features/time_series_clusters.py)   

2. [Feature extraction](https://github.com/YaninaK/sales-forecast/blob/main/notebooks/02_Feature_extraction.ipynb)    
    * [Feature engineering](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/features/build_dataset.py)
    * [Train-validation split](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/data/validation.py)       
    
3. [Baseline model - LSTM](https://github.com/YaninaK/sales-forecast/blob/main/notebooks/03_Baseline_model.ipynb)
    * [Load dataset](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/data/make_dataset.py)
    * [Data preprocessing pipeline](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/models/train.py)
    * [Train test datasets for LSTM](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/data/train_test_datasets.py)
    * [Model architecture](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/models/model_LSTM.py)
    * [Train & save model](https://github.com/YaninaK/sales-forecast/blob/main/scripts/train_save_model.py)
    * [Save artifacts](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/models/save_artifacts.py)
    * [Serialize model](https://github.com/YaninaK/sales-forecast/blob/main/src/sales_forecast/models/serialize.py)
