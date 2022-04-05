#!/usr/bin/env python

#####################
# This command installs the following library, in case the library is already installed, please comment out the command.
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", 'feature_engine'])
#################

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from feature_engine.imputation import RandomSampleImputer
from math import floor


def impute_by_random(data, training_data, col_name):
    imputer = RandomSampleImputer(random_state=16)  # as our seed
    imputer.fit(training_data[col_name].values.reshape(-1, 1))
    return imputer.transform(data[col_name].values.reshape(-1, 1))


def normal_boundaries(col: pd.Series):
    max = col.mean() + 3 * col.std()
    min = col.mean() - 3 * col.std()
    return (min, max)


def skewed_boundaries(col: pd.Series):
    percentile25 = col.quantile(0.25)
    percentile75 = col.quantile(0.75)
    iqr = percentile75 - percentile25
    max = percentile75 + 1.5 * iqr
    min = percentile25 - 1.5 * iqr
    return (min, max)


def outliers_set_nan_iqr(data_col, training_col):
    lower_limit, upper_limit = skewed_boundaries(training_col)
    #print(lower_limit)
    return np.where((data_col >= upper_limit) | (data_col <= lower_limit), np.nan, data_col)


def outliers_set_nan_zscore(data_col, training_col):
    lower_limit, upper_limit = normal_boundaries(training_col)
    return np.where((data_col >= upper_limit) | (data_col <= lower_limit), np.nan, data_col)


def prepare_data(data, training_data):
    data_copy = data.copy()
    training_data_copy = training_data.copy()

    # Transforming blood_type feature into dummy features for OHE but imputing the missing data first:
    data_copy['blood_type'] = impute_by_random(data_copy, training_data_copy, 'blood_type').values
    training_data_copy['blood_type'] = impute_by_random(training_data_copy, training_data_copy, 'blood_type').values
    ohe_blood_type_data = pd.get_dummies(data_copy.blood_type, prefix='blood_type')
    ohe_blood_type_train = pd.get_dummies(training_data_copy.blood_type, prefix='blood_type')

    # Transform symptoms feature into dummy features
    df_symptoms = data_copy['symptoms'].str.get_dummies(sep=';').add_prefix('symptoms_')
    data_copy = pd.concat([data_copy, df_symptoms], axis='columns')
    df_symptoms = training_data_copy['symptoms'].str.get_dummies(sep=';').add_prefix('symptoms_')
    training_data_copy = pd.concat([training_data_copy, df_symptoms], axis='columns')

    # Extract U.S. states (and army region specifiers for military addresses) as features from address 
    data_copy['address_states'] = data_copy['address'].str.slice(start=-8, stop=-6)
    training_data_copy['address_states'] = training_data_copy['address'].str.slice(start=-8, stop=-6)

    # Impute the missing data before transforming into OHE vectors
    data_copy['address_states'] = impute_by_random(data_copy, training_data_copy, 'address_states').values
    training_data_copy['address_states'] = impute_by_random(training_data_copy, training_data_copy,
                                                            'address_states').values
    df_address_states_data = data_copy.address_states.str.get_dummies().add_prefix('address_')
    df_address_states_train = training_data_copy.address_states.str.get_dummies().add_prefix('address_')

    # Impute missing data & transform the sex feature into dummy feature for OHE
    data_copy['sex'] = impute_by_random(data_copy, training_data_copy, 'sex').values
    training_data_copy['sex'] = impute_by_random(training_data_copy, training_data_copy, 'sex').values
    ohe_sex_data = pd.get_dummies(data_copy.sex, prefix='sex')
    ohe_sex_data.drop('sex_F', axis='columns', inplace=True)
    ohe_sex_train = pd.get_dummies(training_data_copy.sex, prefix='sex')
    ohe_sex_train.drop('sex_F', axis='columns', inplace=True)
    training_data_copy['age_groups'] = pd.cut(training_data_copy.age, bins=[0,1,12,19,60,150], labels=['infant', 'child', 'teenager', 'adult', 'senior citizen'])
    training_data_copy['weight_groups'] = pd.cut(training_data_copy.weight, np.arange(0, 136, 5))
    training_data_copy['h_income_groups'] = pd.cut(training_data_copy.household_income, bins=[0,80,178,555,595,793,400_000_000], \
                                              labels=['poverty level', 'low income', 'middle class', 'upper middle class', 'high income', 'highest tax brackets'])
    training_data_copy['sugar_levels_groups'] = pd.cut(training_data_copy.sugar_levels, bins=8)
    training_data_copy['PCR_01_groups'] = pd.cut(training_data_copy.PCR_01, bins=8)
    training_data_copy['PCR_02_groups'] = pd.cut(training_data_copy.PCR_02, bins=8)
    training_data_copy['PCR_03_groups'] = pd.cut(training_data_copy.PCR_03, bins=8)
    training_data_copy['PCR_04_groups'] = pd.cut(training_data_copy.PCR_04, bins=8)
    training_data_copy['PCR_06_groups'] = pd.cut(training_data_copy.PCR_06, bins=8)
    training_data_copy['PCR_07_groups'] = pd.cut(training_data_copy.PCR_07, bins=8)
    training_data_copy['PCR_08_groups'] = pd.cut(training_data_copy.PCR_08, bins=8)
    training_data_copy['PCR_09_groups'] = pd.cut(training_data_copy.PCR_09, bins=8)


    data_copy['age_groups'] = pd.cut(data_copy.age, bins=[0,1,12,19,60,150], labels=['infant', 'child', 'teenager', 'adult', 'senior citizen'])
    data_copy['weight_groups'] = pd.cut(data_copy.weight, np.arange(0, 136, 5))
    data_copy['h_income_groups'] = pd.cut(data_copy.household_income, bins=[0,80,178,555,595,793,400_000_000], \
                                              labels=['poverty level', 'low income', 'middle class', 'upper middle class', 'high income', 'highest tax brackets'])
    data_copy['sugar_levels_groups'] = pd.cut(data_copy.sugar_levels, bins=8)
    data_copy['PCR_01_groups'] = pd.cut(data_copy.PCR_01, bins=8)
    data_copy['PCR_02_groups'] = pd.cut(data_copy.PCR_02, bins=8)
    data_copy['PCR_03_groups'] = pd.cut(data_copy.PCR_03, bins=8)
    data_copy['PCR_04_groups'] = pd.cut(data_copy.PCR_04, bins=8)
    data_copy['PCR_06_groups'] = pd.cut(data_copy.PCR_06, bins=8)
    data_copy['PCR_07_groups'] = pd.cut(data_copy.PCR_07, bins=8)
    data_copy['PCR_08_groups'] = pd.cut(data_copy.PCR_08, bins=8)
    data_copy['PCR_09_groups'] = pd.cut(data_copy.PCR_09, bins=8)
    # First of all, we'll drop the columns which provide zero information:
    data_copy.drop(['job', 'address', 'current_location', 'symptoms'], axis='columns', inplace=True)
    training_data_copy.drop(['job', 'address', 'current_location', 'symptoms'], axis='columns', inplace=True)

    num_cols = ['age', 'weight', 'num_of_siblings', 'happiness_score', 'household_income', 'conversations_per_day',
                'sugar_levels', \
                'sport_activity', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_08',
                'PCR_09', 'PCR_10']

    # clean univariate outliers by replacing with NaN value according to IQR for non normal distributed features:

    n_dist_features = ['PCR_01', 'PCR_02', 'PCR_07']
    for col in [x for x in num_cols if x not in n_dist_features]:
        data_copy[col] = outliers_set_nan_iqr(data_copy[col], training_data_copy[col])
    for col in [x for x in num_cols if x not in n_dist_features]:
        training_data_copy[col] = outliers_set_nan_iqr(training_data_copy[col], training_data_copy[col])

    # clean univariate outliers by replacing with NaN value according to z-score for normal distributed features:
    for col in n_dist_features:
        data_copy[col] = outliers_set_nan_zscore(data_copy[col], training_data_copy[col])
    for col in n_dist_features:
        training_data_copy[col] = outliers_set_nan_zscore(training_data_copy[col], training_data_copy[col])

    # Contextual outlier cleaning for PCR features
    data_copy['PCR_01'] = np.where(data_copy['PCR_01'] >= 1.5, np.nan, \
                                   np.where(data_copy['PCR_01'] <= -1.5, np.nan, data_copy['PCR_01']))
    data_copy['PCR_02'] = np.where(data_copy['PCR_02'] >= 1.5, np.nan, \
                                   np.where(data_copy['PCR_02'] <= -1.5, np.nan, data_copy['PCR_02']))
    data_copy['PCR_04'] = np.where(data_copy['PCR_04'] >= 400, np.nan, \
                                   data_copy['PCR_04'])
    data_copy['PCR_06'] = np.where(data_copy['PCR_06'] <= -11, np.nan, \
                                   data_copy['PCR_06'])
    data_copy['PCR_07'] = np.where(data_copy['PCR_07'] >= 40, np.nan, \
                                   np.where(data_copy['PCR_07'] <= -40, np.nan, data_copy['PCR_07']))
    data_copy['PCR_08'] = np.where(data_copy['PCR_04'] <= -5, np.nan, \
                                   data_copy['PCR_08'])
    data_copy['PCR_09'] = np.where(data_copy['PCR_09'] > 10, np.nan, \
                                   data_copy['PCR_09'])

    training_data_copy['PCR_01'] = np.where(training_data_copy['PCR_01'] >= 1.5, np.nan, \
                                            np.where(training_data_copy['PCR_01'] <= -1.5, np.nan,
                                                     training_data_copy['PCR_01']))
    training_data_copy['PCR_02'] = np.where(training_data_copy['PCR_02'] >= 1.5, np.nan, \
                                            np.where(training_data_copy['PCR_02'] <= -1.5, np.nan,
                                                     training_data_copy['PCR_02']))
    training_data_copy['PCR_04'] = np.where(training_data_copy['PCR_04'] >= 400, np.nan, \
                                            training_data_copy['PCR_04'])
    training_data_copy['PCR_06'] = np.where(training_data_copy['PCR_06'] <= -11, np.nan, \
                                            training_data_copy['PCR_06'])
    training_data_copy['PCR_07'] = np.where(training_data_copy['PCR_07'] >= 40, np.nan, \
                                            np.where(training_data_copy['PCR_07'] <= -40, np.nan,
                                                     training_data_copy['PCR_07']))
    training_data_copy['PCR_08'] = np.where(training_data_copy['PCR_04'] <= -5, np.nan, \
                                            training_data_copy['PCR_08'])
    training_data_copy['PCR_09'] = np.where(training_data_copy['PCR_09'] > 10, np.nan, \
                                            training_data_copy['PCR_09'])

    # Contextual outlier cleaning for non-PCR features
    pcr_cols = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_08', 'PCR_09', 'PCR_10']
    limited_cat_cols = ['conversations_per_day', 'num_of_siblings', 'happiness_score', 'sex', 'symptoms_cough',
                        'symptoms_headache', \
                        'symptoms_low_appetite', 'symptoms_shortness_of_breath', 'symptoms_fever', 'sport_activity',\
                         'blood_type', 'age_groups', 'weight_groups', 'h_income_groups', 'sugar_levels_groups',
                        'PCR_01_groups' \
        , 'PCR_02_groups', 'PCR_03_groups', 'PCR_04_groups', 'PCR_06_groups', 'PCR_07_groups', 'PCR_08_groups',
                        'PCR_09_groups' \
        , 'PCR_05', 'PCR_10', 'address_states']

    for y_feat in [x for x in num_cols if x not in pcr_cols]:
        for x_feat in limited_cat_cols:
            if y_feat == x_feat: continue
            cats_data = data_copy[x_feat].dropna().unique()
            cats_train = training_data_copy[x_feat].dropna().unique()
            num_cats_data = len(cats_data)
            num_cats_train = len(cats_train)
            if num_cats_train == 0:
                continue
            outlier_sum = 0
            for cat in cats_train:
                sub_col = training_data_copy[training_data_copy[x_feat] == cat]
                min, max = skewed_boundaries(sub_col[y_feat])
                outlier_sum += (sub_col[(sub_col[y_feat] >= max) | (sub_col[y_feat] <= min)].shape[0])
            outlier_num_mean = floor(outlier_sum / num_cats_train)
            for cat in cats_data:
                sub_col_data = data_copy.loc[data_copy[x_feat] == cat, :]
                sub_col_train = training_data_copy.loc[training_data_copy[x_feat] == cat, :]
                min, max = skewed_boundaries(sub_col_train[y_feat])
                if sub_col_data[(sub_col_data[y_feat] >= max) | (sub_col_data[y_feat] <= min)].shape[
                    0] <= outlier_num_mean:
                    data_copy.loc[data_copy[x_feat] == cat, y_feat] = outliers_set_nan_iqr(sub_col_data[y_feat],
                                                                                           sub_col_train[y_feat])
                if sub_col_train[(sub_col_train[y_feat] >= max) | (sub_col_train[y_feat] <= min)].shape[
                    0] <= outlier_num_mean:
                    training_data_copy.loc[training_data_copy[x_feat] == cat, y_feat] = outliers_set_nan_iqr(
                        sub_col_train[y_feat], sub_col_train[y_feat])

    # Remove temporary group features from training set
    data_copy.drop(columns=['age_groups', 'weight_groups', 'h_income_groups', 'sugar_levels_groups', 'PCR_01_groups',
                            'PCR_02_groups', 'PCR_03_groups' \
        , 'PCR_04_groups', 'PCR_06_groups', 'PCR_07_groups', 'PCR_08_groups', 'PCR_09_groups'], inplace=True)
    training_data_copy.drop(
        columns=['age_groups', 'weight_groups', 'h_income_groups', 'sugar_levels_groups', 'PCR_01_groups',
                 'PCR_02_groups', 'PCR_03_groups' \
            , 'PCR_04_groups', 'PCR_06_groups', 'PCR_07_groups', 'PCR_08_groups', 'PCR_09_groups'], inplace=True)

    # Replace categorical features with OHE features in training set:
    data_copy = pd.concat([data_copy, ohe_sex_data, ohe_blood_type_data, df_address_states_data], axis='columns')
    data_copy.drop(['sex', 'address_states'], axis=1, inplace=True)
    training_data_copy = pd.concat([training_data_copy, ohe_sex_train, ohe_blood_type_train, df_address_states_train],
                                   axis='columns')
    training_data_copy.drop(['sex', 'address_states'], axis=1, inplace=True)

    # We received an approximate 5% so we'll go with median imputation technique
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')

    # Fit the imputer to the train data
    imputer.fit(training_data_copy['num_of_siblings'].values.reshape(-1, 1))

    # Apply the transformation to the train data
    data_copy['num_of_siblings'] = imputer.transform(data_copy['num_of_siblings'].values.reshape(-1, 1))
    training_data_copy['num_of_siblings'] = imputer.transform(
        training_data_copy['num_of_siblings'].values.reshape(-1, 1))

    median_imp_cols = ['weight', 'household_income']
    random_imp_cols = ['sugar_levels','happiness_score', 'conversations_per_day', 'PCR_06', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_01',
                       'PCR_09', 'PCR_07', 'PCR_10', 'sport_activity']
    should_be_deleted_cause_corr = ['age', 'PCR_05', 'PCR_08', 'pcr_date']

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    for col in median_imp_cols:
        imputer.fit(training_data_copy[col].values.reshape(-1, 1))
        data_copy[col] = imputer.transform(data_copy[col].values.reshape(-1, 1))
        training_data_copy[col] = imputer.transform(training_data_copy[col].values.reshape(-1, 1))

    for col in random_imp_cols:
        data_copy[col] = impute_by_random(data_copy, training_data_copy, col).values
        training_data_copy[col] = impute_by_random(training_data_copy, training_data_copy, col).values

    # Add blood_A_AB feature to data set

    data_copy['blood_A_AB'] = np.where((data_copy['blood_type_A+'] == 1) | (data_copy['blood_type_A-'] == 1) \
                                       | (data_copy['blood_type_AB+'] == 1) | (data_copy['blood_type_AB-'] == 1), 1, 0)
    training_data_copy['blood_A_AB'] = np.where(
        (training_data_copy['blood_type_A+'] == 1) | (training_data_copy['blood_type_A-'] == 1) \
        | (training_data_copy['blood_type_AB+'] == 1) | (training_data_copy['blood_type_AB-'] == 1), 1, 0)

    # Drop features from the data set that we decided not to keep

    to_keep = ['num_of_siblings', 'sugar_levels', 'household_income', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_06', 'PCR_07', 'PCR_10',\
               'blood_type', 'blood_A_AB', 'VirusScore']
    to_drop = [x for x in training_data_copy.columns if x not in to_keep]
    #print(data_copy.columns)
    for drop_it in to_drop:
        data_copy.drop(drop_it, axis='columns', inplace=True)
        training_data_copy.drop(drop_it, axis='columns', inplace=True)

    # Apply normalization to the features

    target_cols = ['blood_type', 'VirusScore']
    z_score_cols = ['PCR_07', 'PCR_10', 'sugar_levels', 'num_of_siblings']
    min_max_cols = [x for x in data_copy.columns if x not in target_cols and x not in z_score_cols]

    z_score_data = data_copy[z_score_cols]
    z_score_training = training_data_copy[z_score_cols]

    min_max_data = data_copy[min_max_cols]
    min_max_training = training_data_copy[min_max_cols]

    targets = data_copy[target_cols]

    scaler = StandardScaler()
    scaler.fit(z_score_training)
    z_score_transformed = pd.DataFrame(scaler.transform(z_score_data), columns=z_score_cols)

    scaler = MinMaxScaler()
    scaler.fit(min_max_training)
    min_max_transformed = pd.DataFrame(scaler.transform(min_max_data), columns=min_max_cols)

    z_score_transformed.reset_index(drop=True, inplace=True)
    min_max_transformed.reset_index(drop=True, inplace=True)
    targets.reset_index(drop=True, inplace=True)

    return pd.concat([min_max_transformed, z_score_transformed, targets], axis='columns')
