import pandas as pd

def check_models(models):
    for model in models:
        test_set = f"datasets/{model}/test.csv"
        test_df = pd.read_csv(test_set)
        other_models = models
        for other_model in other_models:
            train_set = f"datasets/{other_model}/train.csv"
            train_df = pd.read_csv(train_set)
            df_len = len(train_df)
            cond = train_df['sequence'].isin(test_df['sequence'])
            data_cond = train_df[cond]
            if len(data_cond) > 0:
                print(data_cond['sequence'])
            train_df.drop(data_cond.index, inplace = True)
            print(f"{df_len - len(train_df)} overlapping sequences on (test {model} - train {other_model}) pair")

models = ["dprom_complete", "cnnprom_complete", "icnn_complete"]
check_models(models)
models = ["dprom_TATA", "cnnprom_TATA"]
check_models(models)
models = ["dprom_nonTATA", "cnnprom_nonTATA"]
check_models(models)