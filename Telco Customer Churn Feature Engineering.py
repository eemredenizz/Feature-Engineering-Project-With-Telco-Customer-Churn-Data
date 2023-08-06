#####################################################################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING PROJECT (TELCO_CHURN)
#####################################################################################
#####################################
# Gerekli kütüphanelerin yüklenmesi
# Import libraries we need
#####################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

################################
# Genel Bakış
# Overview
################################
df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()
df.isnull().sum()
df["TotalCharges"] = df["TotalCharges"].replace(" ", "nan")
df["TotalCharges"] = df["TotalCharges"].astype(float)
#################################
# num_cols ve cat_cols'ların yakalanması
# Catching num_cols and cat_cols
#################################
for col in df.columns:
    print(col, df[col].nunique())

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

binary_cols = [col for col in df.columns if df[col].nunique() == 2]

#################################
# Hedef - Değişken analizi
# Target - Variable analysis
#################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))

for num in num_cols:
    target_summary_with_num(df, "Churn", num)

####################################
# Aykırı Değerler
# Outliers
####################################

def outlier_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    interquantile_range = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * interquantile_range
    up_limit = quartile3 + 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))


###########################
# Eksik Değerler
# Missing_values
###########################
df.isnull().sum()
df.dropna(inplace=True)
####################################
# Korelasyon Analizi
# Analysis of Corelatıon
####################################

corr = df.corr()
print(corr)

####################################
# Crating New Features
# Yeni Özelliklerin oluşturulması
####################################
df.head()

df["gender"] = df["gender"].replace("Female", 0)
df["gender"] = df["gender"].replace("Male", 1)

df["Partner"] = df["Partner"].replace("No", 0)
df["Partner"] = df["Partner"].replace("Yes", 1)

df["Dependents"] = df["Dependents"].replace("No", 0)
df["Dependents"] = df["Dependents"].replace("Yes", 1)

df["PhoneService"] = df["PhoneService"].replace("No", 0)
df["PhoneService"] = df["PhoneService"].replace("Yes", 1)

df["PaperlessBilling"] = df["PaperlessBilling"].replace("No", 0)
df["PaperlessBilling"] = df["PaperlessBilling"].replace("Yes", 1)

df.loc[df["MonthlyCharges"] > 65, "MonthlyCharges_new"] = "High"

df.loc[df["MonthlyCharges"] <= 65, "MonthlyCharges_new"] = "Low"

df.loc[df["TotalCharges"] > 1700, "TotalCharges_new"] = "High"

df.loc[df["TotalCharges"] <= 1700, "TotalCharges_new"] = "Low"

df.loc[df["Dependents"] + df["Partner"] == 2, "Dependents_x_Partner"] = 1

df.loc[df["Dependents"] + df["Partner"] < 2, "Dependents_x_Partner"] = 0

df.loc[(df["gender"] == 0) & (df["SeniorCitizen"] == 1), "old_female"] = 1

df.loc[(df["gender"] == 0) & (df["SeniorCitizen"] == 0), "old_female"] = 0

df.loc[(df["gender"] == 0) & (df["SeniorCitizen"] == 0), "young_female"] = 1

df.loc[(df["gender"] == 0) & (df["SeniorCitizen"] == 1), "young_female"] = 0

df.loc[(df["gender"] == 1) & (df["SeniorCitizen"] == 1), "old_male"] = 1

df.loc[(df["gender"] == 1) & (df["SeniorCitizen"] == 0), "old_male"] = 0

df.loc[(df["gender"] == 1) & (df["SeniorCitizen"] == 0), "young_male"] = 1

df.loc[(df["gender"] == 1) & (df["SeniorCitizen"] == 1), "young_male"] = 0

df["old_female"] = df["old_female"].fillna(0)
df["young_female"] = df["old_female"].fillna(0)
df["old_male"] = df["old_male"].fillna(0)
df["young_male"] = df["old_male"].fillna(0)

df["tenure_level"] = pd.qcut(df["tenure"], 3, ["low", "normal", "high"])

df.loc[(df["TotalCharges_new"] == "High") & (df["tenure_level"] == "normal"), "Total_charge_X_tenure"] = 5
df.loc[(df["TotalCharges_new"] == "High") & (df["tenure_level"] == "low"), "Total_charge_X_tenure"] = 4
df.loc[(df["TotalCharges_new"] == "Normal") & (df["tenure_level"] == "high"), "Total_charge_X_tenure"] = 5
df.loc[(df["TotalCharges_new"] == "Normal") & (df["tenure_level"] == "normal"), "Total_charge_X_tenure"] = 4
df.loc[(df["TotalCharges_new"] == "Normal") & (df["tenure_level"] == "low"), "Total_charge_X_tenure"] = 3
df.loc[(df["TotalCharges_new"] == "Low") & (df["tenure_level"] == "high"), "Total_charge_X_tenure"] = 4
df.loc[(df["TotalCharges_new"] == "Low") & (df["tenure_level"] == "normal"), "Total_charge_X_tenure"] = 3
df.loc[(df["TotalCharges_new"] == "Low") & (df["tenure_level"] == "low"), "Total_charge_X_tenure"] = 2
df.loc[(df["TotalCharges_new"] == "High") & (df["tenure_level"] == "high"), "Total_charge_X_tenure"] = 6
df.loc[(df["TotalCharges_new"] == "Normal") & (df["tenure_level"] == "high"), "Total_charge_X_tenure"] = 5
df.loc[(df["TotalCharges_new"] == "Low") & (df["tenure_level"] == "high"), "Total_charge_X_tenure"] = 4
df.loc[(df["TotalCharges_new"] == "High") & (df["tenure_level"] == "normal"), "Total_charge_X_tenure"] = 5
df.loc[(df["TotalCharges_new"] == "Normal") & (df["tenure_level"] == "normal"), "Total_charge_X_tenure"] = 4
df.loc[(df["TotalCharges_new"] == "Low") & (df["tenure_level"] == "normal"), "Total_charge_X_tenure"] = 3
df.loc[(df["TotalCharges_new"] == "High") & (df["tenure_level"] == "low"), "Total_charge_X_tenure"] = 4
df.loc[(df["TotalCharges_new"] == "Normal") & (df["tenure_level"] == "low"), "Total_charge_X_tenure"] = 3
df.loc[(df["TotalCharges_new"] == "Low") & (df["tenure_level"] == "low"), "Total_charge_X_tenure"] = 2

df.loc[(df["MonthlyCharges_new"] == "High") & (df["tenure_level"] == "high"), "tenure_x_monthlynew"] = 5
df.loc[(df["MonthlyCharges_new"] == "High") & (df["tenure_level"] == "normal"), "tenure_x_monthlynew"] = 3
df.loc[(df["MonthlyCharges_new"] == "High") & (df["tenure_level"] == "low"), "tenure_x_monthlynew"] = 1
df.loc[(df["MonthlyCharges_new"] == "Low") & (df["tenure_level"] == "high"), "tenure_x_monthlynew"] = 4
df.loc[(df["MonthlyCharges_new"] == "Low") & (df["tenure_level"] == "normal"), "tenure_x_monthlynew"] = 2
df.loc[(df["MonthlyCharges_new"] == "Low") & (df["tenure_level"] == "low"), "tenure_x_monthlynew"] = 0

df["mixed_func"] = df["tenure_x_monthlynew"] + df["Total_charge_X_tenure"]

##################################
# Encoding
##################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

##### Binary Encoding
binary_cols = [col for col in cat_cols if df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)
###### One-Hot Encoding
ohe_cols = [col for col in cat_cols if col not in binary_cols]

def one_hot_encoder(dataframe, ohe_col, drop_first=True):
    dataframe = pd.get_dummies(dataframe,columns= ohe_col, drop_first= drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)

#############################################
# Standart Scaler
#############################################
scaler = StandardScaler()

df = df.replace([np.inf, -np.inf], np.nan).dropna()


df[num_cols] = scaler.fit_transform(df[num_cols])

#######################
#############################################
# 8. Model
#############################################
y = df["Churn"]
X = df.drop(["customerID", "Churn"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)