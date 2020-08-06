import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

df = pd.read_excel('/Users/pf/Desktop/test5.xlsx')
le = preprocessing.LabelEncoder()
df['tr_ac_name'] = le.fit_transform(df['tr_ac_name'])
df['party'] = le.fit_transform(df['party'])
df['deposit_lost'] = le.fit_transform(df['deposit_lost'])

cols = ['tr_ac_name','n_cand','turnout_percentage','enop','party','deposit_lost']
X = df[cols]
y = df['vote_share_percentage']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
result_df = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})

print(reg.score(X_test, y_test))
coeff_df = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)