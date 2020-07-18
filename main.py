from helpers.data_helper import data

X_train, y_train, X_test, y_test, X_val, y_val = data.read_data("TRUNK")
print(len(X_train))
