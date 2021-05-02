# VivaLaData-FinalProject-5-2-21

#Importing Data Sets
import pandas as pd
df1 = pd.read_csv('DisabilityStats.csv')
print(df1)

#Visualizations 
import numpy as np
import matplotlib.pyplot as plt

df3.head()
array = np.array(df3) 
print(array)

#Create pie chart that displays the types of disabilities for Ages 18-64
plt.pie(array[:,1], labels = array[:,0])
plt.title('Types of Disabilities for Ages 18-64')

#Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import statsmodels.api as sm

DataMode = pd.read_csv('DisabilityDataModel.csv')
DataMode.head(27)

df_dummy = pd.get_dummies(DataMode["Disability Type"],drop_first=True)
df_dummy=pd.concat([DataMode,df_dummy],axis=1)
df_dummy.shape

print(DataMode.describe(include ='all'))

X = df_dummy.iloc[:,4:11]
Y = df_dummy.iloc[:,1]

regr = LinearRegression() # Do not use fit_intercept = False if you have removed 1 column after dummy encoding
regr.fit(X, Y)
predicted = regr.predict(X)

print(predicted)

print(DataMode.describe())
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())
