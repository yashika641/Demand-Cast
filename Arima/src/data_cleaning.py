import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df= pd.read_csv(r'C:\Users\palya\Desktop\DemandCast\Demand-Cast\datasets\historical sales data(amul) - Untitled - Sheet 1 (1).csv')
print(df.head())

print(df.shape)
print(df.isnull().sum())
print(df.describe())
print(df.duplicated().sum())

df['Festival_Season'] = df['Festival_Season'].fillna('No Festival')
print(df.isnull().sum())
print(df.head())
df_copy=df.copy()

scaler= MinMaxScaler()
encoder=LabelEncoder()

numerical_cols= df_copy.select_dtypes(include=['float64','boolean','int64']).columns
catgeorical_cols= ['Product_Category', 'Product_Name', 'SKU_Code', 'Region',
                'Festival_Season', 'Weather_Temp', 'Social_Sentiment']
print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", catgeorical_cols)

df_copy[numerical_cols]=scaler.fit_transform(df[numerical_cols])
for col in catgeorical_cols:
    df_copy[col]=encoder.fit_transform(df_copy[col])
df_copy['Date']=pd.to_datetime(df_copy['Date'])
df_copy['Month']=df_copy['Date'].dt.month
df_copy['Year']=df_copy['Date'].dt.year
df_copy['Day']=df_copy['Date'].dt.day
df_copy['Price_Diff'] = df_copy['Price_per_Unit'] - df_copy['Competitor_Price']
# df['Temp_Category'] = pd.cut(df['Weather_Temp'], bins=[-100,15,25,40], labels=['Cold','Mild','Hot'])
# df_copy = df_copy.sort_values(['Product_Name','Date'])
df_copy['Lag_1'] = df_copy.groupby(['Product_Name','Region'])['Units_Sold'].shift(1)
df_copy['Lag_7'] = df_copy.groupby(['Product_Name','Region'])['Units_Sold'].shift(7)

Q1 = df_copy['Units_Sold'].quantile(0.10)
Q3 = df_copy['Units_Sold'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
df_copy['Outlier_Flag'] = ((df_copy['Units_Sold'] < lower_bound) | (df_copy['Units_Sold'] > upper_bound)).astype(int)

# df_copy.drop(['Date'],axis=1,inplace=True)

print(df_copy.head())
print(df_copy.columns)

plt.boxplot(df_copy['Social_Sentiment'])
plt.title("Boxplot to Detect Outliers")
plt.show()
import matplotlib.pyplot as plt

df_copy['Units_Sold'].hist(bins=30)
plt.show()


df_copy.to_csv(r'C:\Users\palya\Desktop\DemandCast\Demand-Cast\datasets\cleaned_data.csv',index=False)