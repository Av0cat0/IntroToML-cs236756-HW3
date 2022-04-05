import prepare
import pandas as pd

if __name__ == '__main__':
    dataset = pd.read_csv("virus_labeled.csv")
    # Selecting samples from the dataset while randomizing the samples
    train_df = dataset.sample(frac=0.8, random_state=16)

    # Selecting all the samples which are not in training_data
    test_df = dataset.drop(train_df.index)

    # Clean training set according to itself without applying normalization
    # train_df_clean_no_norm = prepare_no_norm.prepare_data(train_df, train_df)
    # Clean test set according to the raw training set without applying normalization
    # test_df_clean_no_norm = prepare_no_norm.prepare_data(test_df, train_df)

    # train_df_clean_no_norm.to_csv("train_df_clean_no_norm.csv", index=False)
    # test_df_clean_no_norm.to_csv("test_df_clean_no_norm.csv", index=False)

    # Clean training set according to itself
    train_df_clean = prepare.prepare_data(train_df, train_df)
    # Clean test set according to the raw training set
    test_df_clean = prepare.prepare_data(test_df, train_df)

    train_df_clean.to_csv("train_df_clean.csv", index=False)
    test_df_clean.to_csv("test_df_clean.csv", index=False)
