#!/usr/bin/env python
# coding: utf-8

# In[176]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[177]:


df=pd.read_csv('abalonedata.csv', header=None)


# In[178]:


df.head()


# In[179]:


df.columns=['Sex','Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Rings']


# In[180]:


df.info()


# In[181]:


# remove the error data ( Height=0)
df=df[df['Height']!=0]


# In[182]:


df.describe()


# In[183]:


col         = 'Rings'
conditions  = [ df[col] > 15,(df[col] <=15) &(df[col]>=11),  (df[col] <=10) &(df[col]>=8),(df[col] <=7) &(df[col]>=0)]
choices     = [ 4, 3,2,1 ]
    
df["Class"] = np.select(conditions, choices, default=np.nan)


# In[184]:


df


# In[185]:


df=df.drop('Rings', axis=1)


# In[186]:


sns.pairplot(df, hue='Class')


# In[187]:


sns.heatmap(df.corr(), annot=True) 


# In[188]:


sns.histplot(df, x='Whole_weight', hue='Class',kde='True')


# In[189]:


g = sns.FacetGrid(df, col='Sex', hue="Class", palette="tab20")
g = (g.map(sns.histplot, 'Whole_weight'))


# In[190]:


df.groupby(['Class','Sex'])['Whole_weight'].describe()


# In[191]:


sns.histplot(df, x='Diameter', hue='Class',kde='True')


# In[192]:


g = sns.FacetGrid(df, col='Sex', hue="Class", palette="tab20")
g = (g.map(sns.histplot, 'Diameter'))


# In[193]:


df.groupby(['Class','Sex'])['Diameter'].describe()


# In[194]:


sns.histplot(df, x='Length', hue='Class',kde='True')


# In[195]:


df.groupby(['Class','Sex'])['Length'].describe()


# In[196]:


g = sns.FacetGrid(df, col='Sex', hue="Class", palette="gnuplot2")
g = (g.map(sns.histplot, 'Length'))


# In[197]:


#Investigate the effect of the number of hidden neurons (eg. 5, 10, 15, 20) for a single hidden layer
from keras.regularizers import l2
from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import mean_absolute_error, mean_squared_error,classification_report, confusion_matrix,accuracy_score
from keras.layers import Dropout
import scipy.stats as st


# In[198]:


df['Sex']=df['Sex'].astype('category')
df['Class']=df['Class'].astype('category')


# In[199]:


df['Sex_new'] = df['Sex'].cat.codes 
df['Class_new'] = df['Class'].cat.codes 


# In[200]:


enc = OneHotEncoder() 
enc_data = pd.DataFrame(enc.fit_transform( 
    df[['Sex_new', 'Class_new']]).toarray()) 


# In[201]:


df=df.join(enc_data)


# In[202]:


print(df)


# In[203]:


X=df.drop(['Class','Sex','Class_new'], axis=1).values
y=df['Class_new'].values


# In[204]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[205]:


scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[206]:


#Investigate the effect of the number of hidden neurons (eg. 5, 10, 15, 20) for a single hidden layer  (using SGD) 


# In[207]:


l=[8,10,16,24,32,64,100,128,200,256]
macro_f1=[]
accuracy=[]
for n in l:
    model = keras.Sequential([ 
    layers.Dense(256, activation='relu', input_shape=X_train[0].shape), 
    layers.BatchNormalization(), 
    layers.Dense(int('{}'.format(n)), activation='relu',kernel_initializer='random_normal'), 
    layers.Dropout(0.3), 
    layers.BatchNormalization(), 
    layers.Dense(1, activation='relu')
    ])

    model.compile( 
        loss='mse', 
        optimizer='SGD', 
        metrics=['categorical_accuracy']
    ) 
    model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=128, validation_data=(X_test, y_test)) 
    y_pred=model.predict(X_test)
    predictions = (model.predict(X_test)>0.5).astype("int32")
    report = classification_report(y_test, predictions, output_dict=True)
    macro_f1.append(report['macro avg']['f1-score'])
    accuracy.append(report['accuracy'])

    


# In[208]:


plt.scatter(macro_f1,accuracy) 
plt.xlabel('F1 Score')
plt.ylabel('Accuracy')# the good number of hidden neurons is 256


# In[209]:


macro_f1_3=[]
accuracy3=[]
l=[0.001, 0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
for n in l:
    model = keras.Sequential([ 
    layers.Dense(256, activation='relu', input_shape=X_train[0].shape), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'), 
    layers.Dropout(0.3), 
    layers.BatchNormalization(), 
    layers.Dense(1, activation='relu')
    ])
    
    model.compile( 
        loss='mse', 
        optimizer=keras.optimizers.SGD(learning_rate=float('{}'.format(n))), 
        metrics=['categorical_accuracy']
        ) 
    model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=128, validation_data=(X_test, y_test)) 
    y_pred=model.predict(X_test)
    predictions = (model.predict(X_test)>0.5).astype("int32")
    report = classification_report(y_test, predictions, output_dict=True)
    macro_f1_3.append(report['macro avg']['f1-score'])
    accuracy3.append(report['accuracy'])
        


# In[210]:


plt.plot(l,macro_f1_3)# good learning rate is 0.004
plt.plot(l,accuracy3)
plt.xlabel('learning rate')
plt.ylabel('Rate')
plt.gca().legend(('Accuracy rate','F1 Score'))


# In[211]:


###Investigate the effect on a different number of hidden layers (1, 2) with the optimal number of hidden neurons (from Part 3).


# In[212]:


macro_f1_4_1=[]
accuracy4_1=[]
for n in range(0,11):
    model = keras.Sequential([ 
    layers.Dense(256, activation='relu', input_shape=X_train[0].shape), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'), 
    layers.Dropout(0.3), 
    layers.BatchNormalization(), 
    layers.Dense(1, activation='relu')
    ])

    model.compile( 
        loss='mse', 
        optimizer=keras.optimizers.SGD(learning_rate=0.004), 
        metrics=['categorical_accuracy']
    ) 
    model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=128, validation_data=(X_test, y_test)) 
    y_pred=model.predict(X_test)
    predictions = (model.predict(X_test)>0.5).astype("int32")
    report = classification_report(y_test, predictions, output_dict=True)
    macro_f1_4_1.append(report['macro avg']['f1-score'])
    accuracy4_1.append(report['accuracy'])


# In[213]:


macro_f1_4_2=[]
accuracy4_2=[]
for n in range(0,11):
    model = keras.Sequential([ 
    layers.Dense(256, activation='relu', input_shape=X_train[0].shape), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'), 
    layers.Dropout(0.3), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='relu')
    ])

    model.compile( 
        loss='mse', 
        optimizer=keras.optimizers.SGD(learning_rate=0.004), 
        metrics=['categorical_accuracy']
    ) 
    model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=128, validation_data=(X_test, y_test)) 
    y_pred=model.predict(X_test)
    predictions = (model.predict(X_test)>0.5).astype("int32")
    report = classification_report(y_test, predictions, output_dict=True)
    macro_f1_4_2.append(report['macro avg']['f1-score'])
    accuracy4_2.append(report['accuracy'])


# In[214]:


macro_f1_4_3=[]
accuracy4_3=[]
for n in range(0,11):
    model = keras.Sequential([ 
    layers.Dense(256, activation='relu', input_shape=X_train[0].shape), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'), 
    layers.Dropout(0.3), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='relu')
    ])

    model.compile( 
        loss='mse', 
        optimizer=keras.optimizers.SGD(learning_rate=0.004), 
        metrics=['categorical_accuracy']
    ) 
    model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=128, validation_data=(X_test, y_test)) 
    y_pred=model.predict(X_test)
    predictions = (model.predict(X_test)>0.5).astype("int32")
    report = classification_report(y_test, predictions, output_dict=True)
    macro_f1_4_3.append(report['macro avg']['f1-score'])
    accuracy4_3.append(report['accuracy'])


# In[215]:


import statistics
print(statistics.stdev(accuracy4_1))
print(statistics.stdev(accuracy4_2))  # best model 
print(statistics.stdev(accuracy4_3))
print("\n")

print(statistics.stdev(macro_f1_4_1))
print(statistics.stdev(macro_f1_4_2))
print(statistics.stdev(macro_f1_4_3))


# In[216]:


n=[0,1,2,3,4,5,6,7,8,9,10]
plt.plot(n,macro_f1_4_1) 
plt.plot(n,accuracy4_1)



plt.plot(n,macro_f1_4_2)
plt.plot(n,accuracy4_2)

plt.plot(n,macro_f1_4_3)
plt.plot(n,accuracy4_3)


plt.xlabel('The number of expirements')
plt.ylabel('Rate')
plt.gca().legend(('F1 Score_1','Accuracy_1','F1 Score_2','Accuracy_2','F1 Score_3','Accuracy_3'))


# In[217]:


plt.scatter(macro_f1_4_1,accuracy4_1, color='blue')


# In[218]:


plt.scatter(macro_f1_4_2,accuracy4_2,color='green')


# In[219]:


plt.scatter(macro_f1_4_3,accuracy4_3,color='red')


# In[220]:


#Investigate the effect of Adam and SGD on training and test performance. 
macro_f1_5SGD=[]
accuracy5_SGD=[]
for n in range(0,11):
    model = keras.Sequential([ 
    layers.Dense(256, activation='relu', input_shape=X_train[0].shape), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'), 
    layers.Dropout(0.3), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='relu')
    ])

    model.compile( 
        loss='mse',
        optimizer=keras.optimizers.SGD(learning_rate=0.004), 
        metrics=['categorical_accuracy']
    ) 
    model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=128, validation_data=(X_test, y_test)) 
    y_pred=model.predict(X_test)
    predictions = (model.predict(X_test)>0.5).astype("int32")
    report = classification_report(y_test, predictions, output_dict=True)
    macro_f1_5SGD.append(report['macro avg']['f1-score'])
    accuracy5_SGD.append(report['accuracy'])


# In[221]:


print(accuracy5_SGD)


# In[222]:


macro_f1_5ADAM=[]
accuracy5_ADAM=[]
for n in range(0,11):
    model = keras.Sequential([ 
    layers.Dense(256, activation='relu', input_shape=X_train[0].shape), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'), 
    layers.Dropout(0.3), 
    layers.BatchNormalization(), 
    layers.Dense(256, activation='relu',kernel_initializer='random_normal'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='relu')
    ])

    model.compile( 
        loss='mse', 
        optimizer=keras.optimizers.Adam(learning_rate=0.004), 
        metrics=['categorical_accuracy']
    ) 
    model.fit(X_train, y_train, epochs=200, verbose=1, batch_size=128, validation_data=(X_test, y_test)) 
    y_pred=model.predict(X_test)
    predictions = (model.predict(X_test)>0.5).astype("int32")
    report = classification_report(y_test, predictions, output_dict=True)
    macro_f1_5ADAM.append(report['macro avg']['f1-score'])
    accuracy5_ADAM.append(report['accuracy'])


# In[223]:


plt.scatter(macro_f1_5SGD,accuracy5_SGD,color = 'blue')
plt.scatter(macro_f1_5ADAM,accuracy5_ADAM,color = 'orange')
plt.xlabel('F1 score')
plt.ylabel('Accuracy')
plt.gca().legend(('SGD','ADAM'))


# In[ ]:




