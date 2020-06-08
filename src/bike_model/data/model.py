import numpy as np
import pandas as pd
import datetime as dt
import os
from pathlib import Path
import re
import joblib
import holidays
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
#from bike_model.util import *

ES_holidays = holidays.ES()

def feature_engineering(hour):
    hour = hour.copy #make a copy to save ram; don't modify original csv

    # Office hours rentals
    hour["IsOfficeHour"] = np.where((hour["hr2"]>=9) & (hour["hr2"]<17) & (hour["weekday2"] == 1),1,0)
    hour["IsOfficeHour"] = hour["IsOfficeHour"].astype("category")

    # Daytime rentals
    hour["IsDayTime"] = np.where((hour["hr2"]>=6) & (hour["hr2"] < 22),1, 0)
    hour["IsDayTime"] = hour["IsDayTime"].astype("category")

    # Rush hour rentals
        # Morning
    hour["IsRushMorning"] = np.where((hour["hr2"] >=6)&(hour["hr2"]<10) & (hour["weekday2"] == 1),1,0)
    hour["IsRushMorning"] = hour["IsRushMorning"].astype("category")
        # Evening
    hour["IsRushEvening"] = np.where((hour["hr2"]>=15) & (hour["hr2"]<19) & (hour["weekday2"] == 1),1,0)
hour.head()
