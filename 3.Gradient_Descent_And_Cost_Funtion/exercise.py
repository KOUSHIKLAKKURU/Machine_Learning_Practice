import numpy as np
import pandas as pd
from sklearn import linear_model
import math

def predict_using_sklearn():
    df=pd.read_csv("test_scores.csv")
    reg=linear_model.LinearRegression()
    reg.fit(df[['math']],df.cs)
    return reg.coef_,reg.intercept_

def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=100000
    n=len(x)
    learning_rate=0.0002
    cost_prev=0
    for i in range(iterations):
        y_predicted=m_curr*x+b_curr
        cost=(1/n)*sum([val**2 for val in y-y_predicted])
        md=-(2/n)*sum(x*(y-y_predicted)) # m derivative
        bd=-(2/n)*sum(y-y_predicted) # b derivative
        m_curr=m_curr-learning_rate*md
        b_curr=b_curr-learning_rate*bd
        if math.isclose(cost,cost_prev,rel_tol=1e-20):
            break
        cost_prev=cost
        print("m: {},b: {},cost: {},iteration: {}".format(m_curr,b_curr,cost,i))
    return m_curr,b_curr
if __name__=='__main__':
    df=pd.read_csv("test_scores.csv")
    x=np.array(df.math)
    y=np.array(df.cs)

    m,b=gradient_descent(x,y)
    print("Using gradient descent m: {},b: {}".format(m,b))

    m_sklearn,b_sklearn=predict_using_sklearn()
    print("Using sklearn library m: {},b: {}".format(m_sklearn,b_sklearn))