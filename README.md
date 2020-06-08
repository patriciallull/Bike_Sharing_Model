# Bike sharing prediction model

## Usage

To install the library:
```
$ # pip install .
```

```python
>>> import datetime as dt
>>> from bike_model.model import train_and_persist, predict
>>> train_and_persist() #trains and saves model as model.pkl
>>> predict({
...       "date": dt.datetime(2011,1,1,0,0,0),
...       "weathersit": 1,
...       "temperature_C": 9.84,
...       "feeling_temperature": 14.395,
...       "humidity": 81.0,
...       "windspeed": 0.0,
...})
1
```
