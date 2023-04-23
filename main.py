import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import plotly.express as px



def descript(desc):
    print('################################')
    print(f'{desc}')
    print('################################')

def getLabelDecode(labelEncoder):
    keys = labelEncoder.classes_
    values = labelEncoder.transform(labelEncoder.classes_)
    dictionary = dict(zip(keys, values))
    return dictionary

def main():
    # df = pd.read_csv('/kaggle/input/salaly-prediction-for-beginer/Salary Data.csv')
    df = pd.read_csv('Salary Data.csv')
    df.head()
    descript('Original data')
    print(df.head(), '\n')


    # https://www.geeksforgeeks.org/dealing-with-rows-and-columns-in-pandas-dataframe/
    df.dropna(axis=0, inplace=True) # Drop rows which contain missing values.

    genderLabel = LabelEncoder()
    df['Gender'] = genderLabel.fit_transform(df['Gender'])
    #df["Gender"] = df["Gender"].replace({ "Female":0, "Male":1})

    educationLabel = LabelEncoder()
    df['Education Level'] = educationLabel.fit_transform(df['Education Level'])

    jobLabel = LabelEncoder()
    df['Job Title'] = jobLabel.fit_transform(df['Job Title'])
    
    descript('label decode')
    print(getLabelDecode(genderLabel), '\n')
    print(getLabelDecode(educationLabel), '\n')
    print(getLabelDecode(jobLabel), '\n')


    df.head()
    descript('after LabelEncoder data')
    print(df.head(), '\n')
    
    descript('data numbers')
    print(len(df), '\n')

    Y = df['Salary']
    X = df.drop(['Salary'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    descript('data shape')
    print('x_train: ', x_train.shape)
    print('x_test: ', x_test.shape)
    print('y_train: ', y_train.shape)
    print('y_test: ', y_test.shape, '\n')

    # https://p61402.github.io/2019/06/12/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%B6%93%E5%85%B8%E6%BC%94%E7%AE%97%E6%B3%95%E5%AF%A6%E4%BD%9C-Linear-Regression/
    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # y(x) = W0 + W1x1 + W2x2 + W3x3 + W4x4 + W5x5
    w_0 = model.intercept_
    w_n = model.coef_
    descript('parameter')
    print('Interception : ', w_0)
    print('Coeficient : ', w_n, '\n')

    accuracy = metrics.r2_score(y_pred, y_test)
    descript('result')
    print('Accuracy: ' + str(accuracy * 100) + '%')

    
    fig = plt.figure(figsize=(8,8), dpi=100)
    plt.title('Years of Experience  VS  Salary')
    plt.xlabel('Years of Experience', {'fontsize':15,'color':'red'})
    plt.ylabel('Salary', {'fontsize':15,'color':'green'})
    plt.scatter(x_test['Years of Experience'], y_test, c='blue', label='grouth')
    plt.scatter(x_test['Years of Experience'], y_pred, c='red', label='predict')
    plt.legend(
        loc='best',
        fontsize=10,
        shadow=True,
        facecolor='#ccc',
        edgecolor='#000',
        title='label',
        title_fontsize=10)
    plt.savefig('docs/test.jpg')


    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    plt.suptitle('test dataset (Salary)')    # main title
    ax[0][0].set_title('Age')  # sub title
    ax[1][0].set_title('Gender')  # sub title
    ax[0][1].set_title('Education Level')  # sub title
    ax[1][1].set_title('Years of Experience')  # sub title

    ax[0][0].scatter(x_test['Age'], y_test, c='blue', label='grouth')
    ax[1][0].scatter(x_test['Gender'], y_test, c='blue', label='grouth')
    ax[0][1].scatter(x_test['Education Level'], y_test, c='blue', label='grouth')
    ax[1][1].scatter(x_test['Years of Experience'], y_test, c='blue', label='grouth')

    ax[0][1].text(0.95, 0.01, f'{getLabelDecode(educationLabel)}',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax[0][1].transAxes,
        color='green', fontsize=10)
    ax[1][0].text(0.95, 0.01, f'{getLabelDecode(genderLabel)}',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax[1][0].transAxes,
        color='green', fontsize=10)
    plt.savefig('docs/test2.jpg')



    dataFrameTest = x_test
    dataFrameTest['Salary'] = y_pred

    fig = px.scatter(dataFrameTest, x="Years of Experience", y="Salary", color=None
                        , hover_data=['Age','Education Level', 'Job Title', 'Years of Experience', 'Salary'])
    fig.write_image("docs/fig1.svg")
    fig.write_html("docs/demo.html")



if __name__ == '__main__':
    main()
