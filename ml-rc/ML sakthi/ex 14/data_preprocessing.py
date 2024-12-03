import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("diabetes.csv")
print(df.head())
df.info()
print("Missing values:\n", df.isnull().sum())
print("Descriptive statistics:\n", df.describe())
fig, axs = plt.subplots(len(df.columns), 1, dpi=95, figsize=(7, 17))
for i, col in enumerate(df.columns):
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_ylabel(col)
plt.tight_layout()
plt.show()
def remove_outliers(data, column):
    q1, q3 = np.percentile(data[column], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
clean_data = df.copy()
for col in ['Insulin', 'Pregnancies', 'Age', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']:
    clean_data = remove_outliers(clean_data, col)
corr = clean_data.corr()
plt.figure(dpi=130)
sns.heatmap(corr, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
plt.pie(clean_data['Outcome'].value_counts(), labels=['Diabetes', 'Not Diabetes'], autopct='%.f%%', shadow=True)
plt.title('Outcome Proportionality')
plt.show()
X = clean_data.drop(columns=['Outcome'])
Y = clean_data['Outcome']
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_minmax = scaler.fit_transform(X)
scaler = StandardScaler()
rescaledX_standard = scaler.fit_transform(X)
print("Min-Max Scaled Data:\n", rescaledX_minmax[:5])
print("Standard Scaled Data:\n", rescaledX_standard[:5])
