from prefect import task, flow
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score


@task
def create_x_y():
    x = np.random.rand(1,1000)[0]
    x= x.reshape(-1,1)
    y = 2*x + 3*x**2 + np.random.randn()
    y = y.reshape(-1,1)
    X_train,X_val,y_train,y_val = train_test_split(x,y,test_size=0.22,random_state=33)
    return X_train,y_train,X_val,y_val

@task(retries=3)
def linreg(X_train,y_train,X_val,y_val):
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"rmse{rmse}")
    return rmse


@flow
def on2(log_prints=True):
    X_train,y_train,X_val,y_val = create_x_y()
    rmse= linreg(X_train,y_train,X_val,y_val)
    print(f"rmse{rmse}")


if __name__ == "__main__":
    on2()
