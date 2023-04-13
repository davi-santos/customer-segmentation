'''
* IMPORTANT NOTES:
*
* This project aims to make a customer segmentation of a grocery.
* The raw dataset should be downloaded on kaggle on this link:
* https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
* and placed on ../data/raw/ folder
*
* There are two jupyter notebooks that contains some statistical analysis,
* but it is not necessary to run or see then in order to run this file. All
* you need to do is download the data and place it on the data folder.
*
* In this file, we will:
* - load the raw data;
* - create new features based on the data;
* - format the data in the correct type;
* - remove outliers;
* - format the data in a way machine learning models can understand;
*
* So, let's get started
'''
import pandas as pd
import numpy as np

def month_diff(first_date, second_date):
    return 12 * (first_date.year - second_date.year) + (first_date.month - second_date.month)

# LOAD THE DATA
# --------------------------------------
customers_dataframe = pd.read_csv('../../data/raw/marketing_campaign.csv', sep='\t')
# --------------------------------------

# CHECK DATA
# --------------------------------------

# Show first rows, see if was correctly loaded
# customers_dataframe.head()

# check info
# customers_dataframe.info()

# check if each row is a unique customer
# customers_dataframe['ID'].unique().shape

# So, each row is a unique customer. 
# The dataset is in the correct format
# --------------------------------------


# FORMAT THE DATA
# --------------------------------------
# -Dt_Customer type is not correct; 
# -Income has a few null values;

# Correct DT_Customer type
customers_dataframe['Dt_Customer'] = pd.to_datetime(customers_dataframe['Dt_Customer'], dayfirst=True)

# Check number of null values
# customers_dataframe.isna().sum()

# Check dataframe number of lines
# customers_dataframe.shape[0]

# There are 24 null rows out of 2240. This is just 1.07%
# of the total dataset. So, we'll just remove them

# Remove null values
customers_dataframe.dropna(inplace=True)
# --------------------------------------

# FEATURE ENGINEERING
# --------------------------------------
# FEATURE 1 - Age
# Consider current date as the most recent buying date
current_year = customers_dataframe['Dt_Customer'].max().year
customers_dataframe['Age'] = customers_dataframe['Year_Birth'].apply(lambda x: current_year-x)

# FEATURE 2 - Number of children
customers_dataframe['Children'] = customers_dataframe['Kidhome'] + customers_dataframe['Teenhome']

# FEATURE 3 - Total members
# has a partner or not dict
marital_status_dict = {
    'Single': 'Alone',
    'Together': 'Pair',
    'Married': 'Pair',
    'Divorced': 'Alone',
    'Widow': 'Alone',
    'Absurd': 'Alone',
    'Alone': 'Alone',
    'YOLO': 'Alone',
}
# create new column about having a partner or not
customers_dataframe['Marital_Status_new'] = customers_dataframe['Marital_Status'].apply(lambda x: marital_status_dict[x])

# create total members feature
customers_dataframe['TotalMembers'] = customers_dataframe['Children'] +\
                                        customers_dataframe['Marital_Status_new'].replace({'Alone': 1, 'Pair': 2})

# FEATURE 3 - Total spent
customers_dataframe['TotalSpent'] = customers_dataframe['MntWines'] + customers_dataframe['MntFruits'] +\
                                    customers_dataframe['MntMeatProducts'] + customers_dataframe['MntFishProducts'] +\
                                    customers_dataframe['MntSweetProducts'] + customers_dataframe['MntGoldProds']

# FEATURE 4 - MEMBER FOR N YEARS
customers_dataframe['Dt_Customer']

# FEATURE 5 - Number of months a customer has enrolment with the grocery
current_date = customers_dataframe['Dt_Customer'].max()

customers_dataframe['n_months_customer'] = customers_dataframe['Dt_Customer'].\
                                            apply(lambda dt_customer: month_diff(current_date, dt_customer))
# --------------------------------------

# TRANSFORM COLUMN VALUES
# --------------------------------------
# Simplify values and transform them to numbers

# check Education values
# customers_dataframe['Education'].unique()

# create new education feature
education_dict = {'Graduation': 1,
                  'PhD': 2,
                  'Master': 2,
                  'Basic': 0,
                  '2n Cycle': 0}

customers_dataframe['education_level'] = customers_dataframe['Education'].apply(lambda x: education_dict[x])

# 
marital_status_dict = {'Alone': 0,
                       'Pair': 1}
customers_dataframe['is_pair'] = customers_dataframe['Marital_Status_new'].apply(lambda x: marital_status_dict[x])
# --------------------------------------

# REMOVE OUTLIERS

# Check stats
# customers_dataframe.describe()

# Some people in this dataset is very old
# and some has a really high income. For now,
# We'll remove them

# Age less than 90
customers_dataframe = customers_dataframe[customers_dataframe['Age'] < 90]

# Income less than 300k
customers_dataframe = customers_dataframe[customers_dataframe['Income'] < 300000]


# SELECT COLUMNS

columns = ['Age',
           'education_level',
           'Children', 
           'is_pair', 
           'TotalMembers',
           'Income', 
           'n_months_customer', 
           'Recency', 
           'MntWines', 
           'MntFruits',
           'MntMeatProducts', 
           'MntFishProducts', 
           'MntSweetProducts',
           'MntGoldProds',
           'TotalSpent',
           'NumDealsPurchases', 
           'NumWebPurchases',
           'NumCatalogPurchases', 
           'NumStorePurchases', 
           'NumWebVisitsMonth']

final_df = customers_dataframe[columns].copy()

# UPDATE COLUMN NAMES TO SIMPLER NAMES
new_column_name = {'Age': 'age',
                   'education_level': 'education',
                   'Children': 'children', 
                   'is_pair': 'isPair', 
                   'TotalMembers': 'familySize',
                   'Income': 'income',
                   'n_months_customer': 'customerFor', 
                   'Recency': 'recency', 
                   'MntWines': 'wines', 
                   'MntFruits': 'fruits',
                   'MntMeatProducts': 'meat', 
                   'MntFishProducts': 'fish', 
                   'MntSweetProducts': 'sweet',
                   'MntGoldProds': 'gold',
                   'TotalSpent': 'totalSpent',
                   'NumDealsPurchases': 'dealsPurchases', 
                   'NumWebPurchases': 'webPurchases',
                   'NumCatalogPurchases': 'catPurchases', 
                   'NumStorePurchases': 'storePurchases', 
                   'NumWebVisitsMonth': 'webVisitsMonth'}          

final_df.rename(columns=new_column_name, inplace = True)

# So, we'll save the created features for the next process
final_df.to_pickle('../../data/processed/01_built_features.pkl')