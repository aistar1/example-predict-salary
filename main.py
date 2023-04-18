import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import plotly.express as px


# df = pd.read_csv('/kaggle/input/salaly-prediction-for-beginer/Salary Data.csv')
df = pd.read_csv('Salary Data.csv')
df.head()
print(df.head(),'\n')


# https://www.geeksforgeeks.org/dealing-with-rows-and-columns-in-pandas-dataframe/
df.dropna(axis=0, inplace=True) # Drop rows which contain missing values.

df["Gender"] = df["Gender"].replace({ "Female":0, "Male":1})
# genderLabel = LabelEncoder()
# df['Gender'] = genderLabel.fit_transform(df['Gender'])

educationLabel = LabelEncoder()
df['Education Level'] = educationLabel.fit_transform(df['Education Level'])

jog_label = LabelEncoder()
df['Job Title'] = jog_label.fit_transform(df['Job Title'])

df.head()
print(df.head(),'\n')

print(len(df),'\n')

Y = df['Salary']
X = df.drop(['Salary'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)


# https://p61402.github.io/2019/06/12/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%B6%93%E5%85%B8%E6%BC%94%E7%AE%97%E6%B3%95%E5%AF%A6%E4%BD%9C-Linear-Regression/
model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
# y(x) = W0 + W1x1 + W2x2 + W3x3 + W4x4 + W5x5
w_0 = model.intercept_
w_n = model.coef_
print('Interception : ', w_0)
print('Coeficient : ', w_n)

accuracy = metrics.r2_score(y_pred, y_test)
print('Accuracy: ' + str(accuracy * 100) + '%')



plt.scatter(x_test['Years of Experience'], y_test)
plt.savefig('docs/test.jpg')




dataFrameTest = x_test
dataFrameTest['Salary'] = y_pred

fig = px.scatter(dataFrameTest, x="Years of Experience", y="Salary", color=None
                     , hover_data=['Age','Education Level', 'Job Title', 'Years of Experience', 'Salary'])
fig.write_image("docs/fig1.png")
fig.write_html("docs/demo.html")
