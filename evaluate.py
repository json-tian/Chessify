import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("training_data.csv", nrows=100)
data = data[["result", "diff"]]
# result: 1 = white wins, 0 = black wins
# diff: rating of white - rating of black

# To print out first few entries: print(data.head())

predict = "result"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Finding highest accuracy within trails
trails = 100
highest = 0
for i in range(trails):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > highest:
        best = acc
    with open("chessmodel.pickle", "wb") as f:
        pickle.dump(linear, f)

pickle_in = open("chessmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Displaying data points on a graph
p = "diff"
style.use("ggplot")
pyplot.scatter(data[p], data["result"])
pyplot.xlabel("Rating Difference")
pyplot.ylabel("Result")
pyplot.show()
