import tensorflow as tf
from tensorflow.python.framework import ops
from Preprocessor import Preprocessor
import matplotlib.pyplot as plt
import numpy as np
import math


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')

    return X, Y


def initialize_parameters():

    W1 = tf.get_variable("W1", [25, 3072], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [30, 25], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [30, 1], initializer=tf.zeros_initializer())
    # W3 = tf.get_variable("W3", [50, 25], initializer=tf.contrib.layers.xavier_initializer())
    # b3 = tf.get_variable("b3", [50, 1], initializer=tf.zeros_initializer())
    Wf = tf.get_variable("Wf", [10, 30], initializer=tf.contrib.layers.xavier_initializer())
    bf = tf.get_variable("bf", [10, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  # "W3": W3,
                  # "b3": b3,
                  "Wf": Wf,
                  "bf": bf}

    return parameters


def forward_propagation(X, parameters, keep_prob):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # W3 = parameters['W3']
    # b3 = parameters['b3']
    Wf = parameters['Wf']
    bf = parameters['bf']

    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    A1 = tf.nn.dropout(A1, keep_prob=keep_prob)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    A2 = tf.nn.dropout(A2, keep_prob=keep_prob)
    # Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
    # A3 = tf.nn.relu(Z3)  # A3 = relu(Z3)
    # A3 = tf.nn.dropout(A3, keep_prob=keep_prob)
    Zf = tf.add(tf.matmul(Wf, A2), bf)  # Z3 = np.dot(W3,Z2) + b3
    return Zf


def compute_cost(Zf, Y):

    logits = tf.transpose(Zf)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0009,
          num_epochs=1000, minibatch_size=64, print_cost=True):

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    keep_prob = tf.placeholder(tf.float32)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Zf = forward_propagation(X, parameters, keep_prob=keep_prob)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Zf, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            validation_cost = 0
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob:0.8})

                epoch_cost += minibatch_cost / num_minibatches
            minibatch_validation_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test, keep_prob: 0.8})
            validation_cost += minibatch_validation_cost
            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                train_summary(epoch, epoch_cost, validation_cost, Zf, X, Y, X_train, Y_train, X_test, Y_test, keep_prob)
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        saver.save(sess, 'model')
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Zf), tf.argmax(Y))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob:1}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, keep_prob:1}))

        return parameters


def train_summary(epoch, epoch_cost, validation_cost, Zf, X, Y, X_train, Y_train, X_test, Y_test, keep_prob):

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Zf), tf.argmax(Y))
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1})
    test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1})

    if epoch == 0:
        print("Loop" + '\t' + "Train Loss" + '\t' + "Train Acc %" + '\t' + "Test Loss" + '\t' + "Test Acc %")
    print("%i \t \t %f \t %f \t %f \t %f" % (epoch, epoch_cost, train_accuracy, validation_cost, test_accuracy))



def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')

    one_hot_matrix = tf.one_hot(labels, C, axis=0)

    sess = tf.Session()

    one_hot = sess.run(one_hot_matrix)

    sess.close

    return one_hot


p = Preprocessor()
X_train, labels_train, X_test, labels_test = p.load_data()
Y_train = one_hot_matrix(labels=labels_train, C=10)
Y_test = one_hot_matrix(labels=labels_test, C=10)

X_train = X_train.T/255
X_test = X_test.T/255

model(X_train, Y_train, X_test, Y_test)



