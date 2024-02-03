import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocessing():
    # Load the dataset
    data = pd.read_excel("Dry_Bean_Dataset.xlsx")

    # Handle missing data (replace missing values with the mean of the column)
    data['MinorAxisLength'].fillna(data['MinorAxisLength'].mean(), inplace=True)

    # Split the dataset into features (X) and the target variable (y)
    X = data.drop(columns=['Class'])  # Features
    y = data['Class']  # Target variable

    # Scale the numeric features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))

    # One Hot Encoding
    y = pd.get_dummies(y, dtype=int)

    data_set = pd.concat([X, y], axis=1, join="inner")
    return data_set, scaler


def data_extraction_and_splitting(data_set):
    # 1st Class shuffling then splitting:
    _c1_data = data_set[data_set['BOMBAY'] == 1]  # should be 50
    _c1_data = pd.DataFrame(_c1_data)
    _X = _c1_data.iloc[:, 0:-1]
    _Y = _c1_data.iloc[:, -1]
    _x_train, _x_test, _y_train, _y_test = train_test_split(_X, _Y, train_size=30, test_size=20, shuffle=True,
                                                            random_state=13)
    _c1_training = pd.concat([_x_train, _y_train], axis=1, join="inner")
    _c1_testing = pd.concat([_x_test, _y_test], axis=1, join="inner")

    # 2nd Class shuffling then splitting:
    _c2_data = data_set[data_set['CALI'] == 1]  # should be 50
    _X = _c2_data.iloc[:, 0:-1]
    _Y = _c2_data.iloc[:, -1]
    _x_train, _x_test, _y_train, _y_test = train_test_split(_X, _Y, train_size=30, test_size=20, shuffle=True,
                                                            random_state=13)
    _c2_training = pd.concat([_x_train, _y_train], axis=1, join="inner")
    _c2_testing = pd.concat([_x_test, _y_test], axis=1, join="inner")

    # 3rd Class shuffling then splitting:
    _c3_data = data_set[data_set['SIRA'] == 1]  # should be 50
    _X = _c3_data.iloc[:, 0:-1]
    _Y = _c3_data.iloc[:, -1]
    _x_train, _x_test, _y_train, _y_test = train_test_split(_X, _Y, train_size=30, test_size=20, shuffle=True,
                                                            random_state=13)
    _c3_training = pd.concat([_x_train, _y_train], axis=1, join="inner")
    _c3_testing = pd.concat([_x_test, _y_test], axis=1, join="inner")

    # gathering training_data,testing_data from both classes:
    _training_data = pd.concat([_c1_training, _c2_training, _c3_training])
    _testing_data = pd.concat([_c1_testing, _c2_testing, _c3_testing])

    # generating x_train,x_test,y_train,y_test
    x_train = np.array(_training_data.iloc[:, 0:5])  # 90[30 from each]
    y_train = np.array(_training_data.iloc[:, 5:8])  # 90[30 from each]
    x_test = np.array(_testing_data.iloc[:, 0:5])  # 60[20 from each]
    y_test = np.array(_testing_data.iloc[:, 5:8])  # 60[20 from each]

    return x_train, x_test, y_train, y_test
