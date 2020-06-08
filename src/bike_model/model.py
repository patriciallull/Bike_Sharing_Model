import numpy as np
import pandas as pd
import datetime as dt
import os
from pathlib import Path
import re
import joblib
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor

from bike_model.util import read_data, get_season, get_model_path

US_holidays = calendar().holidays()


def feature_engineering(hour):
    hour = hour.copy()  # make a copy to save ram; don't modify original csv

    # Office hours rentals
    hour["IsOfficeHour"] = np.where(
        (hour["hr2"] >= 9) & (hour["hr2"] < 17) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsOfficeHour"] = hour["IsOfficeHour"].astype("category")

    # Daytime rentals
    hour["IsDayTime"] = np.where((hour["hr2"] >= 6) & (hour["hr2"] < 22), 1, 0)
    hour["IsDayTime"] = hour["IsDayTime"].astype("category")

    # Rush hour rentals
    # Morning
    hour["IsRushMorning"] = np.where(
        (hour["hr2"] >= 6) & (hour["hr2"] < 10) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsRushMorning"] = hour["IsRushMorning"].astype("category")
    # Evening
    hour["IsRushEvening"] = np.where(
        (hour["hr2"] >= 15) & (hour["hr2"] < 19) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsRushEvening"] = hour["IsRushEvening"].astype("category")

    # Seasons rentals
    hour["IsHighSeason"] = np.where((hour["season2"] == 3), 1, 0)
    hour["IsHighSeason"] = hour["IsHighSeason"].astype("category")

    # Bins for temperature and humidity (5)
    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    hour["temp_binned"] = pd.cut(hour["temp2"], bins).astype("category")
    hour["hum_binned"] = pd.cut(hour["hum2"], bins).astype("category")

    return hour


def preprocess(hour):
    hour = hour.copy()

    # Create duplicate columns for feature_engineering
    hour["hr2"] = hour["hr"]
    hour["season2"] = hour["season"]
    hour["temp2"] = hour["temp"]
    hour["hum2"] = hour["hum"]
    hour["weekday2"] = hour["weekday"]

    # Convert to datetime
    hour["dteday"] = pd.to_datetime(hour["dteday"])

    # Convert data type to category or float
    int_hour = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    for col in int_hour:
        hour[col] = hour[col].astype("category")

    # Log1p
    logws = round(skew(np.log1p(hour.windspeed)), 4)

    # Sqrt of count
    sqrtws = round(skew(np.sqrt(hour.windspeed)), 4)
    hour["windspeed"] = np.log1p(hour.windspeed)

    # Log of count
    logcnt = round(skew(np.log(hour.cnt)), 4)

    # Sqrt of count
    sqrtcnt = round(skew(np.sqrt(hour.cnt)), 4)
    hour["cnt"] = np.sqrt(hour.cnt)

    hour = feature_engineering(hour)

    # Drop duplicated columns used for feature_engineering
    hour = hour.drop(columns=["hr2", "season2", "temp2", "hum2", "weekday2"])

    return hour


def dummify(hour, known_columns=None):
    hour = pd.get_dummies(hour)
    if known_columns is not None:
        for col in known_columns:
            if col not in hour.columns:
                hour[col] = 0
        hour = hour[known_columns]

    return hour


def split_train_test(hour):
    hour = hour.copy()

    # Split
    hour_train, hour_test = hour.iloc[0:15211], hour.iloc[15212:17379]
    train = hour_train.drop(columns=["dteday", "casual", "atemp", "registered"])
    test = hour_test.drop(columns=["dteday", "casual", "atemp", "registered"])

    # Separate features from target on train set
    train_X = train.drop(columns = ["cnt"], axis=1)
    train_y = train["cnt"]
    
    # Separate features from target on the test set
    test_X = test.drop(columns=["cnt"], axis=1)
    test_y = test["cnt"]

    return train_X, test_X, train_y, test_y


def train_random_forest(hour):
    hour = hour.copy()

    hour_d = pd.get_dummies(hour)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    hour_d.columns = [
        regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
        for col in hour_d.columns.values
    ]

    hour_d = hour_d.select_dtypes(exclude="category")

    hour_d_train_x, _, hour_d_train_y, _, = split_train_test(hour_d)

    rf = RandomForestRegressor(
        max_depth=40,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=42,
    )

    rf.fit(hour_d_train_x, hour_d_train_y)
    return rf


def postprocess(hour):
    hour = hour.copy()

    hour.columns = hour.columns.str.replace("[\[\]\<]", "_")
    return hour


def train_and_persist(model_dir=None, hour_path=None):
    # set default values to None in case no dir param is specified
    hour = read_data(hour_path)
    hour = preprocess(hour)
    hour = dummify(hour)

    # Model
    model = train_random_forest(hour)

    model_path = get_model_path(model_dir)

    joblib.dump(model, model_path)


def get_input_dict(parameters):
    hour_original = read_data()
    base_year = pd.to_datetime(hour_original["dteday"]).min().year

    date = parameters["date"]

    is_holiday = date in US_holidays
    is_weekend = date.weekday() in (5, 6)

    row = pd.Series(
        {
            "dteday": date.strftime("%Y-%m-%d"),
            "season": get_season(date),
            "yr": date.year - base_year,
            "mnth": date.month,
            "hr": date.hour,
            "holiday": 1 if is_holiday else 0,
            "weekday": (date.weekday() + 1) % 7,
            "workingday": 0 if is_holiday or is_weekend else 1,
            "weathersit": parameters["weathersit"],
            "temp": parameters["temperature_C"] / 41.0,
            "atemp": parameters["feeling_temperature_C"] / 50.0,
            "hum": parameters["humidity"] / 100.0,
            "windspeed": parameters["windspeed"] / 67.0,
            "cnt": 1,  # Dummy, unused for prediction
        }
    )

    dummiefied_original = dummify(preprocess(hour_original))

    df = pd.DataFrame([row])
    df = preprocess(df)
    df = dummify(df, dummiefied_original.columns)
    df = postprocess(df)

    df = df.drop(columns=["dteday", "atemp", "casual", "registered", "cnt"])

    assert len(df) == 1

    return df.iloc[0].to_dict()


def predict(parameters, model_dir=None):
    """ Returns the prediction for the parameters specified
    """
    model_path = get_model_path(model_dir)
    if not os.path.exists(model_path):
        train_and_persist(model_dir)

    model = joblib.load(model_path)

    input_dict = get_input_dict(parameters)
    X_input = pd.DataFrame([pd.Series(input_dict)])

    result = model.predict(X_input)

    return int(result ** 2)  # to undo np.sqrt(hour["count"])
