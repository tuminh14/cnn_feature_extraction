from helpers.data_helper import data

X_train, y_train, X_test, y_test = data.read_data("TRUNK")
print(len(X_train))
