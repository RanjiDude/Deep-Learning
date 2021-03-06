import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('Deep-Learning/Datasets/train_catvnoncat.h5', "r") # edit path as per location of .h5 file
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # training set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # training set labels

    test_dataset = h5py.File('Deep-Learning/Datasets/test_catvnoncat.h5', "r") # edit path as per location of .h5 file
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# pre-process the datasets
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# standardize the image dataset by dividing every row by 255 (max pixel value)
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# sigmoid function
def sigmoid(z):

    s = 1/(1+np.exp(-z))

    return s

def initialize_parameters(dim):

    w = np.zeros((dim,1))
    b = 0

    assert (w.shape == (dim,1))

    return w, b

# forward and backward propogation yields the parameter changes dw and db
def propagate(w, b, X, Y):

    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)                         # sigmoid activation function for 0 hidden layer binary classification
    cost = (-1/m)*np.sum(Y*np.log(A)+((1-Y)*np.log(1-A)))

    dw = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    grads = {"dw" : dw, "db" : db}

    return grads, cost

#training the model in order to minimize the cost function and finding optimal values for w and b parameters
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs

# prediction function
def predict(w, b, X):

    m = X.shape[1]

    Y_prediction = np.zeros((1,m))
    w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] <= 0.5:
            Y_prediction[0,i] = 0
        elif A[0, i] > 0.5:
            Y_prediction[0,i] = 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost = False):

    w, b = initialize_parameters(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = True)

    w = params["w"]
    b = params["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.01, print_cost = False)
