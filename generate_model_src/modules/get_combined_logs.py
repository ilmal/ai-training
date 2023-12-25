import pandas as pd

def get_combined_logs():
    # get the logs from each model
    model_logs_save_path = "/results/model_logs/"
    model_logs = []
    for i in range(1, 11):
        model_logs.append(pd.read_csv(f"{model_logs_save_path}model_{i}.txt", sep=" - ", engine="python"))
    # print(model_logs[0].head())
    # print(model_logs[0].tail())

    # combine the logs into one dataframe
    combined_logs = pd.concat(model_logs, axis=0)
    # print(combined_logs.head())
    # print(combined_logs.tail())

    # save the combined logs
    combined_logs.to_csv(f"{model_logs_save_path}combined_logs.csv", index=False)

    # get the logs from each model
    model_logs_save_path = "/results/model_logs/"
    model_logs = []
    for i in range(1, 11):
        model_logs.append(pd.read_csv(f"{model_logs_save_path}model_{i}.txt", sep=" - ", engine="python"))
    # print(model_logs[0].head())
    # print(model_logs[0].tail())

    # combine the logs into one dataframe
    combined_logs = pd.concat(model_logs, axis=0)
    # print(combined_logs.head())
    # print(combined_logs.tail())

    # save the combined logs
    combined_logs.to_csv(f"{model_logs_save_path}combined_logs.csv", index=False)