import numpy as np
import pandas as pd
from datetime import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
# https://www.kdnuggets.com/2020/08/5-different-ways-load-data-python.html


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
plt.style.use('seaborn-colorblind')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 10),
          'figure.titlesize': 'x-large',
          'figure.dpi': 150,
         'axes.labelsize': 'x-large',
         'axes.titlesize': 'x-large',
         'xtick.labelsize': 'x-large',
         'ytick.labelsize': 'x-large',
         'savefig.bbox': 'tight'}

plt.rcParams.update(params)
# plt.rcParams.keys() ### Use to check the available parameters

def datagen(train_size, test_size, folds):
    yield 1, 0, train_size
    for i in range(2, folds+1):
        yield i, (train_size+test_size)*(i-1), ((train_size+test_size)*(i-1))+train_size

        
def seasonal_arima(df, train_size, test_size, folds, order=(0,1,0),
                   seasonal_order = (0,1,0,24), *args):
    rmse=[]
    preds=[]
    indexes = []
    for ii, tr, te in datagen(train_size, test_size, folds): 
            cv_train, cv_test = df.iloc[tr:tr+train_size], df.iloc[te:te+test_size]
            print(f'Fold {ii}: train index ({tr}), test_index ({te})')
            #print(cv_train)    

    ##################################################################################
    ###### Function to compute the rolling window predictions of the Spot price ######
    ##################################################################################
            mod = sm.tsa.statespace.SARIMAX(cv_train, order = order,
                                               seasonal_order = seasonal_order,
                                               enforce_stationarity = False,
                                               enforce_invertibility  = False)
            results = mod.fit()

            """ Useful to dig into a specific model
            It shows the summary statistics, the residuals
            and how it compares to the normal distribution
            """
            #print(results.summary())
            # line plot of residuals
            #residuals = pd.DataFrame(results.resid)
            #residuals.plot()
            #plt.show()
            # density plot of residuals
            #residuals.plot(kind='kde')
            #plt.show()
            # summary stats of residuals
            #print(residuals.describe())
            #break
            predictions = results.forecast(steps = test_size)
            true_values = cv_test.values
            preds.append(predictions)
            rmse.append(np.sqrt(mean_squared_error(true_values, predictions)))
            indexes.append([te, te+test_size])

    print("RMSE: {}".format(np.mean(rmse)))
    return preds, rmse, indexes

def calculate_mape(actual, predicted) -> float:
  
    # Convert actual and predicted
    # to numpy array data type if not already
    if not all([isinstance(actual, np.ndarray),
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)
  
    # Calculate the MAPE value and return
    return np.mean(np.abs((
      actual - predicted) / actual)) * 100


def plot_preds(tdat, preds, indexes, train_size, folds, order, *args):
    print(f'The SARIMAX order is {order}')
    for ii in range(folds):
        fig = plt.figure(figsize=(20,10))
        plt.plot(tdat[(indexes[ii][0]-train_size):indexes[ii][0]], label = "True price train")
        plt.plot(tdat[indexes[ii][0]:indexes[ii][1]], label = "True price test")
        plt.plot(preds[ii], label = "Predicted price")
        plt.title(f'Forecast from {tdat[indexes[ii][0]:indexes[ii][1]].index[0]} to {tdat[indexes[ii][0]:indexes[ii][1]].index[-1]}')
        plt.legend()
        fig.savefig(f'Plots/Pred {order} Fold {ii+1}.jpg')
        plt.show()
        
        
def step_rmse(preds, tdat, folds, *args):
    preds_idx=[]
    preds_values=[]
    for i in range(folds):
        preds_idx.extend(preds[i].index)
        preds_values.extend(preds[i].values)    
    preds_df = pd.DataFrame(data=preds_values, index = preds_idx)
    preds_df.reset_index(inplace = True)
    preds_df.columns = ["datetime", "Predicted Price"]
    preds_df["hour"] = preds_df["datetime"].dt.hour
    preds_df["dayofweek"] = preds_df["datetime"].dt.weekday
    true_test = tdat.reset_index()
    preds_df = preds_df.merge(tdat, how = "left", on = "datetime")
    # Square root of the incremental mean of the difference
    preds_df["RMSE"] = np.sqrt(((preds_df["Predicted Price"] - preds_df.price)**2).expanding().mean()) 
    return preds_df


def revenue(df, folds, mape, test_size, **args):
    
    """ Function to calculate price difference and impact
        of the MAPE on the smar charging-related revenue
    """
    
    
    ######## Get Price difference ######## """Actual Price - Predicted Price"""
    df["pricediff"] = df.price - df["Predicted Price"]

    ######## Get MAPE Column ########
    mape_cols = [str(c) for c in range(folds)]
    mapedf= pd.DataFrame(columns = mape_cols, index = range(test_size))
    mapedf=mapedf.assign(**dict(zip(mape_cols, mape)))
    mape_list = []
    for _, ro in mapedf.T.iterrows():
        mape_list.extend(ro.values)
    df["mape"] = mape_list

    ######## Get Price Loss ########
    df["price_loss"]= df.pricediff * df.mape / 100

    return df

