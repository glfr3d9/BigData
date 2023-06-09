import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('heart_failure_clinical_records_dataset.csv') #Загрузка датасета в датафрейм
df = df.drop(columns =
['anaemia','diabetes','high_blood_pressure','sex','smoking','time','DEATH_EVENT'])
# Сброс ограничений на число столбцов
# pd.set_option('display.max_columns', None)
# Сброс ограничений на количество символов в записи
# pd.set_option('display.max_colwidth', None)# Установите для отображения самой большой линии
# print(df)

data = df.to_numpy(dtype='float')
n_bins = 20
fig, axs = plt.subplots(2,3)
axs[0, 0].hist(df['age'].values, bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(df['creatinine_phosphokinase'].values, bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(df['ejection_fraction'].values, bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(df['platelets'].values, bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(df['serum_creatinine'].values, bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(df['serum_sodium'].values, bins = n_bins)
axs[1, 2].set_title('serum_sodium')

print('До стандартизации')
print(np.var(data))
print(np.std(data))

# scaler = preprocessing.StandardScaler().fit(data[:150,:])
scaler = preprocessing.StandardScaler().fit(data[:299,:])
data_scaled = scaler.transform(data)
fig, axs = plt.subplots(2,3)
axs[0, 0].hist(data_scaled[:,0], bins = n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_scaled[:,1], bins = n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_scaled[:,2], bins = n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_scaled[:,3], bins = n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_scaled[:,4], bins = n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_scaled[:,5], bins = n_bins)
axs[1, 2].set_title('serum_sodium')

print('После стандартизации')
print(np.var(data_scaled))
print(np.std(data_scaled))

print('mean_,var_')
print(scaler.mean_)
print(scaler.var_)

min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data_min_max_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_min_max_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_min_max_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_min_max_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_min_max_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_min_max_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

print('min_max')
print(min_max_scaler.data_min_)
print(min_max_scaler.data_max_)

max_abs_scaler = preprocessing.MaxAbsScaler().fit(data)
data_max_abs_scaled = max_abs_scaler.transform(data)
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data_max_abs_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_max_abs_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_max_abs_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_max_abs_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_max_abs_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_max_abs_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

r_scaler = preprocessing.RobustScaler().fit(data)
data_r_scaler = r_scaler.transform(data)
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data_r_scaler[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_r_scaler[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_r_scaler[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_r_scaler[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_r_scaler[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_r_scaler[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

func = preprocessing.MinMaxScaler(feature_range=(-5, 10))
data_func_scaler = func.fit_transform(data)
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data_func_scaler[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_func_scaler[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_func_scaler[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_func_scaler[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_func_scaler[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_func_scaler[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100, random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data_quantile_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_quantile_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_quantile_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_quantile_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_quantile_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_quantile_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

quantile_transformer = preprocessing.QuantileTransformer(n_quantiles = 100,output_distribution = 'normal',random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data_quantile_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_quantile_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_quantile_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_quantile_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_quantile_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_quantile_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

power_scaler = preprocessing.PowerTransformer().fit(data)
data_power_scaler = power_scaler.transform(data)
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data_power_scaler[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_power_scaler[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_power_scaler[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_power_scaler[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_power_scaler[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_power_scaler[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')

est = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4], encode='ordinal', strategy='uniform').fit(data)
data_quantile_scaled = est.transform(data)
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data_quantile_scaled[:, 0], bins=3)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_quantile_scaled[:, 1], bins=4)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_quantile_scaled[:, 2], bins=3)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_quantile_scaled[:, 3], bins=10)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_quantile_scaled[:, 4], bins=2)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_quantile_scaled[:, 5], bins=4)
axs[1, 2].set_title('serum_sodium')
print('Диапазоны')
print(np.histogram_bin_edges(data))
plt.show()
# plt.show()
