import numpy as np
import pandas as pd
from datetime import datetime as dt

# https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
# https://www.kdnuggets.com/2020/08/5-different-ways-load-data-python.html


def read_spot_prices(url_base, empty_rows = 5, usecols="A,B,C,D,F:Z", **args):
    """ Function to read the Nordpool data, which comes in a pivottable - like format
    
        Hour  Hour  Hour  Hour  ... Hour
    Day Price Price Price Price ... Price 
    
    """
    col_names = ["date", "01", "02", "03", "04", "05", "06",
                 "07", "08", "09", "10", "11", "12",
                 "13", "14", "15", "16", "17", "18",
                 "19", "20", "21", "22", "23", "00"]
    cols2use = usecols
    df = pd.read_excel(url_base, usecols=cols2use, names=col_names)
    df = df.iloc[empty_rows:]
    return df

def data_processing(url_base, univ=True):
    """
    Takes NordPool's data in DataFrame format and corrects the misnaming of the hours,
    Assigns the correct index,
    Changes the price units to €/KWh (from €/MWh),
    Normalizes the prices,
    Adds datetime features to help explicitly infer the seasonality.
    """
    
    # Get data into a DataFrame
    currency = 'eur'
    filetype = ".xls"
    years = ["17", "18", "19", "20", "21", "22"] #Years we want to get historical data
    for y in years:
        if y == years[0]: prices = read_spot_prices(url_base+currency+str(y)+filetype)
        else: 
            df2 = read_spot_prices(url_base+currency+str(y)+filetype)
            prices = prices.append(df2)
    prices.dropna(subset = ["date"], how = "any", axis = 0, inplace=True)
    prices.drop_duplicates(subset=["date"], keep = "last", ignore_index = True, inplace=True)
    price_dat = prices.drop("date", axis = 1)
    dat = []
    # Solve the misnaming of Nordpool (24h to 00h format)
    for ii, ro in price_dat.iterrows():
        dat.extend(ro.values)
    
    # Remove first 23 hours since day 1st jan 2017 is ordered incorrectly
    dat = dat[23:] 
    # Keep the values for the dates that have passed by, remove future prices
    dat = dat[:45505] 
    
    # Give DataFrame format with date range
    date_rng = pd.date_range(start='1/02/2017', end='3/13/2022', freq='H')
    df = pd.DataFrame(data = dat, columns = ["price"])
    
    # Add datetime features
    df["datetime"] = date_rng
    df = df[:-1]
    df.interpolate(inplace = True) # To check if ok use """df[1990:2000]"""
    
    if univ == True:
        df["weekday"] = df["datetime"].dt.weekday
        df["week"] = df["datetime"].dt.week
        df["day"] = df["datetime"].dt.day
    
    return df

