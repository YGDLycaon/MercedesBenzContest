import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

first_df=pd.read_csv('train.csv')
y_train=first_df['y']
#----------------------------------------------------------------------
print (first_df.info())
#print(first_df.describe())
#--------------------------------------------------------------------
#4209 rows X 378 columns
# columna "y" es el target 
# columnas "X0 a X8" son categóricas (no hay "X7")
# 8 columnas categoricas , 369 columnas int , 1 columna float
#---------------------------------------------------------------------

missing=first_df.isnull()
#print (missing)
#sns.heatmap(missing,cmap='magma')
#plt.show()
# al parecer no hay valores faltantes en el dataset
#-----------------------------------------------------------------------------
catData=first_df[['X0','X1','X2','X3','X4','X5','X6','X8']]

#print (catData.head(2))
#print (catData['X0'].value_counts())
print ('X0:')
print (catData['X0'].unique())
#sns.countplot(x='X0',data=catData)
#plt.show()
print ('X1:')
print (catData['X1'].unique())
#print (catData['X1'].value_counts())
#sns.countplot(x='X1',data=catData)
#plt.show()
print ('X2:')
print (catData['X2'].unique())
#sns.countplot(x='X2',data=catData)
#plt.show()
print ('X3:')
print (catData['X3'].unique())
#print (catData['X3'].value_counts())
#sns.countplot(x='X3',data=catData)
#plt.show()
print ('X4:')
print (catData['X4'].unique())
#print (catData['X4'].value_counts())
#sns.countplot(x='X4',data=catData)
#plt.show()
print ('X5:')
print (catData['X5'].unique())
#print (catData['X5'].value_counts())
#sns.countplot(x='X5',data=catData)
#plt.show()
print ('X6:')
print (catData['X6'].unique())
#sns.countplot(x='X6',data=catData)
#plt.show()
print ('X8:')
print (catData['X8'].unique())
#sns.countplot(x='X8',data=catData)
#plt.show()

#---------------------------------------------------------------
# categoriza de x0 a x8
x_0=pd.get_dummies(catData['X0'])
x_1=pd.get_dummies(catData['X1'])
x_2=pd.get_dummies(catData['X2'])
x_3=pd.get_dummies(catData['X3'])
x_4=pd.get_dummies(catData['X4'])
x_5=pd.get_dummies(catData['X5'])
x_6=pd.get_dummies(catData['X6'])
x_8=pd.get_dummies(catData['X8'])

dummies=[x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_8]
df_dummies=pd.concat(dummies,axis=1)
#print (df_dummies)
#------------------------------------------------------------------

first_df.drop(['ID','y','X0','X1','X2','X3','X4','X5','X6','X8'],axis=1,inplace=True)
df=pd.concat([df_dummies,first_df],axis=1)

#----Reduccion de dimensiones-----

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)

from sklearn.decomposition import PCA

pca=PCA(n_components=20)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
print ('xpca:\n',x_pca)
print (np.shape(x_pca))
x_train=pd.DataFrame(x_pca)
df_comp=pd.DataFrame(pca.components_)
plt.figure(figsize=(12,8))
sns.heatmap(df_comp,cmap='plasma')
plt.show()


#---------leer test.csv----------------------------------------------------------------------

prueba=pd.read_csv('test.csv')
prueba.drop(['ID'],axis=1,inplace=True)
catDataTest=prueba[['X0','X1','X2','X3','X4','X5','X6','X8']]
#---------------------------------------------------------------
# categoriza de x0 a x8
xTest_0=pd.get_dummies(catDataTest['X0'])
xTest_1=pd.get_dummies(catDataTest['X1'])
xTest_2=pd.get_dummies(catDataTest['X2'])
xTest_3=pd.get_dummies(catDataTest['X3'])
xTest_4=pd.get_dummies(catDataTest['X4'])
xTest_5=pd.get_dummies(catDataTest['X5'])
xTest_6=pd.get_dummies(catDataTest['X6'])
xTest_8=pd.get_dummies(catDataTest['X8'])

dummiesTest=[xTest_0,xTest_1,xTest_2,xTest_3,xTest_4,xTest_5,xTest_6,xTest_8]
df_dummiesTest=pd.concat(dummiesTest,axis=1)

prueba.drop(['X0','X1','X2','X3','X4','X5','X6','X8'],axis=1,inplace=True)
df_test=pd.concat([df_dummiesTest,prueba],axis=1)
#------------------------PCA a test--------------------------------

scaler1=StandardScaler()
scaler1.fit(df_test)
scaled_data1=scaler1.transform(df_test)
pca1=PCA(n_components=20)
pca1.fit(scaled_data1)
x_pca1=pca1.transform(scaled_data1)
print (np.shape(x_pca1))
x_test=pd.DataFrame(x_pca1)

#-------------------regresion lineal para prediccion------------------------
# ya no es necesario realizar cross validation, el dataset ya está dividio en train y test

from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge

#lm=LinearRegression()
#lm=Ridge()
lm=BayesianRidge()
lm.fit(x_train,y_train)

print ('Coef: ',lm.coef_)

predictions=lm.predict(x_test)

print (predictions)
print (np.shape(predictions))
pred=pd.DataFrame(predictions)
#pred.to_csv(path_or_buf= 'C:\\Users\\Lycaon\\Documents\\CIC\Mercedes Benz\\', sep=',')
pred.to_csv('predictions5.csv', sep=',')
