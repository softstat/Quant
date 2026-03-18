import pandas as pd

def lag_features(df, lag=1):
    return df.shift(lag)

def get_rebalance_dates(price_df, freq="M"):
    return price_df.resample(freq).last().index

def point_in_time_universe(universe_df, date):
    return universe_df[universe_df["date"] <= date]["ticker"].unique()