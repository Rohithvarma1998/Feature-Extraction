
# coding: utf-8

# In[234]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pandas import read_csv
from pandas import concat
from pandas import DataFrame

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

import scipy.stats


# In[235]:


glucose_series_pat1 = pd.read_csv("./data/CGMSeriesLunchPat1.csv")
glucose_series_pat1.head()

glucose_series_pat2 = pd.read_csv("./data/CGMSeriesLunchPat2.csv")
glucose_series_pat2.head()

glucose_series_pat3 = pd.read_csv("./data/CGMSeriesLunchPat3.csv")
glucose_series_pat3.head()

glucose_series_pat4 = pd.read_csv("./data/CGMSeriesLunchPat4.csv")
glucose_series_pat4.head()

glucose_series_pat5 = pd.read_csv("./data/CGMSeriesLunchPat5.csv")
glucose_series_pat5.head()


# In[236]:


time_stamp = pd.read_csv("./data/CGMDatenumLunchPat1.csv")
time_stamp = time_stamp.applymap(lambda i : pd.to_datetime(i - 719529, unit='D'))
time_stamp.head(6)


# In[237]:


#arranging the cgm_series data in order of time
columns = glucose_series_pat1.columns.tolist()
columns = columns[::-1]
glucose_series_sorted_pat1 = glucose_series_pat1[columns]

columns = glucose_series_pat2.columns.tolist()
columns = columns[::-1]
glucose_series_sorted_pat2 = glucose_series_pat2[columns]

columns = glucose_series_pat3.columns.tolist()
columns = columns[::-1]
glucose_series_sorted_pat3 = glucose_series_pat3[columns]

columns = glucose_series_pat4.columns.tolist()
columns = columns[::-1]
glucose_series_sorted_pat4 = glucose_series_pat4[columns]

columns = glucose_series_pat5.columns.tolist()
columns = columns[::-1]
glucose_series_sorted_pat5 = glucose_series_pat5[columns]


# In[238]:


#data cleaning
def dataCleaning(df):
    glucose_cols = df.columns
    velo_data_temp = pd.DataFrame(columns = glucose_cols)
    velo_data = velo_data_temp.append(df, ignore_index = False)
    df.fillna(df.mean(),inplace=True)
    return velo_data, df

velo_data_pat1, cleaned_CGM_pat1 = dataCleaning(glucose_series_sorted_pat1)
velo_data_pat2, cleaned_CGM_pat2 = dataCleaning(glucose_series_sorted_pat2)
velo_data_pat3, cleaned_CGM_pat3 = dataCleaning(glucose_series_sorted_pat3)
velo_data_pat4, cleaned_CGM_pat4 = dataCleaning(glucose_series_sorted_pat4)
velo_data_pat5, cleaned_CGM_pat5 = dataCleaning(glucose_series_sorted_pat5)


# In[239]:


#feature extraction: Feature1:- Autocorrelation
def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    return result[:result.size//2]

def df_autocorrelation(df):
    rows,columns = df.shape
    corr_matrix=[]
    for i in range(rows):
        result = autocorrelation(df.iloc[i])
        reverse_result = np.flipud(result)
        corr_matrix.append(reverse_result/reverse_result.max())
    return corr_matrix

feature_one = df_autocorrelation(cleaned_CGM_pat1)
auto_corr_pat1 = pd.DataFrame(feature_one)

feature_one = df_autocorrelation(cleaned_CGM_pat2)
auto_corr_pat2 = pd.DataFrame(feature_one)

feature_one = df_autocorrelation(cleaned_CGM_pat3)
auto_corr_pat3 = pd.DataFrame(feature_one)

feature_one = df_autocorrelation(cleaned_CGM_pat4)
auto_corr_pat4 = pd.DataFrame(feature_one)

feature_one = df_autocorrelation(cleaned_CGM_pat5)
auto_corr_pat5 = pd.DataFrame(feature_one)


# In[240]:


#autocorrelation analysis
def plot_curve(df):
    rows = df.shape[0]
    for i in range(rows):
        plt.plot(df.iloc[i,::3])
    plt.show()

plt.plot(auto_corr_pat1)
plt.show()
plt.plot(auto_corr_pat2)
plt.show()
plt.plot(auto_corr_pat3)
plt.show()
plt.plot(auto_corr_pat4)
plt.show()
plt.plot(auto_corr_pat5)
plt.show()


# In[241]:


#rate of change of glucose level every 2 intervals
def row_velocity(record):
    result=[]
    cols = record.shape[0]
    for j in range(cols - 1):
        calc_velocity = (record[j+1] - record[j])/5
        result.append(calc_velocity)
    return result

def velocity_of_df(df):
    row,cols = df.shape
    vel_matrix=[]
    for i in range(row):
        row_vel = row_velocity(df.iloc[i])
        vel_matrix.append(row_vel)
    return vel_matrix
        
feature_two = velocity_of_df(cleaned_CGM_pat1)
velocity_pat1 = pd.DataFrame(feature_two)

feature_two = velocity_of_df(cleaned_CGM_pat2)
velocity_pat2 = pd.DataFrame(feature_two)

feature_two = velocity_of_df(cleaned_CGM_pat3)
velocity_pat3 = pd.DataFrame(feature_two)

feature_two = velocity_of_df(cleaned_CGM_pat4)
velocity_pat4 = pd.DataFrame(feature_two)

feature_two = velocity_of_df(cleaned_CGM_pat5)
velocity_pat5 = pd.DataFrame(feature_two)

plt.plot(velocity_pat1)
plt.show()
plt.plot(velocity_pat2)
plt.show()
plt.plot(velocity_pat3)
plt.show()
plt.plot(velocity_pat4)
plt.show()
plt.plot(velocity_pat5)
plt.show()


# In[242]:


def row_fft(record):
    result=[]
    cols = record.shape[0]
    result = scipy.fftpack.fft(record)
    result = np.log(np.abs(result))
    return result

def fft_of_df(df):
    row,cols = df.shape
    fft_matrix=[]
    for i in range(row):
        row_fftransform = row_fft(df.iloc[i])
        fft_matrix.append(row_fftransform)
    return fft_matrix

feature_three = fft_of_df(cleaned_CGM_pat1)
fft_pat1 = pd.DataFrame(feature_three)

feature_three = fft_of_df(cleaned_CGM_pat2)
fft_pat2 = pd.DataFrame(feature_three)

feature_three = fft_of_df(cleaned_CGM_pat3)
fft_pat3 = pd.DataFrame(feature_three)

feature_three = fft_of_df(cleaned_CGM_pat4)
fft_pat4 = pd.DataFrame(feature_three)

feature_three = fft_of_df(cleaned_CGM_pat5)
fft_pat5 = pd.DataFrame(feature_three)

plt.plot(fft_pat1)
plt.show()
plt.plot(fft_pat2)
plt.show()
plt.plot(fft_pat3)
plt.show()
plt.plot(fft_pat4)
plt.show()
plt.plot(fft_pat5)
plt.show()


# In[243]:


#Moving average:ma
def row_ma(record):
    result=[]
    cols = record.shape[0]
    windowSize = 3
    for i in range(cols-windowSize+1):
        mean = np.mean(record[i:i+windowSize])
        result.append(mean)
    return result

def ma_of_df(df):
    row,cols = df.shape
    ma_matrix=[]
    for i in range(row):
        row_movavg = row_ma(df.iloc[i])
        ma_matrix.append(row_movavg)
    return ma_matrix

feature_four = ma_of_df(cleaned_CGM_pat1)
ma_pat1 = pd.DataFrame(feature_four)

feature_four = ma_of_df(cleaned_CGM_pat2)
ma_pat2 = pd.DataFrame(feature_four)

feature_four = ma_of_df(cleaned_CGM_pat3)
ma_pat3 = pd.DataFrame(feature_four)

feature_four = ma_of_df(cleaned_CGM_pat4)
ma_pat4 = pd.DataFrame(feature_four)

feature_four = ma_of_df(cleaned_CGM_pat5)
ma_pat5 = pd.DataFrame(feature_four)

plt.plot(ma_pat1)
plt.show()
plt.plot(ma_pat2)
plt.show()
plt.plot(ma_pat3)
plt.show()
plt.plot(ma_pat4)
plt.show()
plt.plot(ma_pat5)
plt.show()


# In[244]:


#feature matrix
df1 = auto_corr_pat1.T
df2 = velocity_pat1.T
df3 = fft_pat1.T
df4 = ma_pat1.T
featureMatrix_pat1 = df1.append([df2,df3,df4], ignore_index=True)
featureMatrix_pat1 = featureMatrix_pat1.T


df1 = auto_corr_pat2.T
df2 = velocity_pat2.T
df3 = fft_pat2.T
df4 = ma_pat2.T
featureMatrix_pat2 = df1.append([df2,df3,df4], ignore_index=True)
featureMatrix_pat2= featureMatrix_pat2.T

df1 = auto_corr_pat3.T
df2 = velocity_pat3.T
df3 = fft_pat3.T
df4 = ma_pat3.T
featureMatrix_pat3 = df1.append([df2,df3,df4], ignore_index=True)
featureMatrix_pat3 = featureMatrix_pat3.T

df1 = auto_corr_pat4.T
df2 = velocity_pat4.T
df3 = fft_pat4.T
df4 = ma_pat4.T

featureMatrix_pat4 = df1.append([df2,df3,df4], ignore_index=True)
featureMatrix_pat4 = featureMatrix_pat4.T

df1 = auto_corr_pat5.T
df2 = velocity_pat5.T
df3 = fft_pat5.T
df4 = ma_pat5.T

featureMatrix_pat5 = df1.append([df2,df3,df4], ignore_index=True)
featureMatrix_pat5 = featureMatrix_pat5.T

featureMatrix_pat1.head()


# In[245]:


def pca_analysis(df):
    scaled_df=StandardScaler().fit_transform(df)
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(scaled_df)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2',
                                                                      'principal component 3','principal component 4',
                                                                      'principal component 5'])
    #a=pca.components_
    #print(a)
    #variance=pca.explained_variance_
    return principalDf, pca

DF_pat1,pca_1 = pca_analysis(featureMatrix_pat1)
featureMatrix_pat2.fillna(0, inplace = True)
DF_pat2,pca_2 = pca_analysis(featureMatrix_pat2)
featureMatrix_pat3.fillna(0, inplace = True)
DF_pat3,pca_3 = pca_analysis(featureMatrix_pat3)
featureMatrix_pat4.fillna(0, inplace = True)
DF_pat4,pca_4 = pca_analysis(featureMatrix_pat4)
featureMatrix_pat5.fillna(0, inplace = True)
DF_pat5,pca_5 = pca_analysis(featureMatrix_pat5)


# In[246]:


def barplot(pca):
    plt.figure()
    print(pca.explained_variance_)
    plt.bar(range(5), pca.explained_variance_)
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# In[247]:


barplot(pca_1)
barplot(pca_2)
barplot(pca_3)
barplot(pca_4)
barplot(pca_5)


# In[248]:


plt.plot(DF_pat1["principal component 1"],'*')
plt.show()
plt.plot(DF_pat1["principal component 2"],'*')
plt.show()
plt.plot(DF_pat1["principal component 3"],'*')
plt.show()
plt.plot(DF_pat1["principal component 4"],'*')
plt.show()
plt.plot(DF_pat1["principal component 5"],'*')
plt.show()


# In[249]:


plt.plot(DF_pat2["principal component 1"],'*')
plt.show()
plt.plot(DF_pat2["principal component 2"],'*')
plt.show()
plt.plot(DF_pat2["principal component 3"],'*')
plt.show()
plt.plot(DF_pat2["principal component 4"],'*')
plt.show()
plt.plot(DF_pat2["principal component 5"],'*')
plt.show()


# In[250]:


plt.plot(DF_pat3["principal component 1"],'*')
plt.show()
plt.plot(DF_pat3["principal component 2"],'*')
plt.show()
plt.plot(DF_pat3["principal component 3"],'*')
plt.show()
plt.plot(DF_pat3["principal component 4"],'*')
plt.show()
plt.plot(DF_pat3["principal component 5"],'*')
plt.show()


# In[251]:


plt.plot(DF_pat4["principal component 1"],'*')
plt.show()
plt.plot(DF_pat4["principal component 2"],'*')
plt.show()
plt.plot(DF_pat4["principal component 3"],'*')
plt.show()
plt.plot(DF_pat4["principal component 4"],'*')
plt.show()
plt.plot(DF_pat4["principal component 5"],'*')
plt.show()


# In[252]:


plt.plot(DF_pat5["principal component 1"],'*')
plt.show()
plt.plot(DF_pat5["principal component 2"],'*')
plt.show()
plt.plot(DF_pat5["principal component 3"],'*')
plt.show()
plt.plot(DF_pat5["principal component 4"],'*')
plt.show()
plt.plot(DF_pat5["principal component 5"],'*')
plt.show()

