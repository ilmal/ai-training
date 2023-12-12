import pandas as pd
import os

# Define a generator function that takes a list of file pairs and a batch size
def data_generator(data_dir, batch_size):
  
    # Get dates
    dates = [e.split(".")[0] for e in os.listdir(f'{data_dir}labels/')]
    
    for date in dates:

        # Load the features and labels
        dfx = pd.read_csv(f'{data_dir}data/{date}.csv', index_col=0).dropna()
        dfy = pd.read_csv(f'{data_dir}labels/{date}.csv', index_col=0).dropna()

        if dfx.shape[1] < 3095:
            raise Exception("THE DATA: ", f"{date}.csv", "is incorrect")

        # Find common tickers
        common_tickers = list(set(dfx.index) & set(dfy.index))

        # Create new dataframes with only common tickers
        dfx_common = dfx.loc[common_tickers]
        dfy_common = dfy.loc[common_tickers]

        # Ensure both dataframes have the same order of tickers
        dfx_common = dfx_common.reindex(dfy_common.index)

        # drop ticker for insertion into model
        dfx_common.reset_index(drop=True, inplace=True)
        dfy_common.reset_index(drop=True, inplace=True)

        num_rows = dfy.shape[0]

        # Loop over the batches
        for i in range(0, num_rows, batch_size):

            dfx_common = dfx_common.loc[i:i+batch_size]
            dfy_common = dfy_common.loc[i:i+batch_size]

            # Yield the batch of features and labels
            yield dfx_common, dfy_common

