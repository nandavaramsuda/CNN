import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



def one_hot_encode(labels):
    n_labels = len(labels)
    # will return total unique value in
    u_nique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,u_nique_labels))
    one_hot_encode[np.arange(n_labels),labels] = 1
    return one_hot_encode


# loading data from csv to machine
df = pd.read_csv("/home/gaurav/AI/Deep_learning/data/training.csv")
print(df.head())

# selecting features and label for training
X = df[df.columns[0:60]].values
y = df[df.columns[60]]
print("x value :{}".format(X))
print("y value :{}".format(y))


# label and onehot encoding for y
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encode(y)
print(y.shape)

# shuffle data
X, Y = shuffle(X, Y, random_state = 1)

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.1, random_state = 1)

# defining important parameters
learning_rate = 0.01
epochs = 1000
n_dim = X.shape[1]
n_classes = 2

# define hidden layers and no of neurons on these layers
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

# defining placeholders for features and labels
x = tf.placeholder(tf.float32,[None,n_dim])
y_ = tf.placeholder(tf.float32,[None,n_classes])


# defining model
def model(x,weights,biases):

    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['h1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['h2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['h3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['h4'])
    layer_4 = tf.nn.relu(layer_4)

    layer_out = tf.matmul(layer_4,weights['out']) + biases['out']
    return layer_out


# defining weight
weights = {

    'h1' : tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
    'h2' : tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3' : tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'h4' : tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4,n_classes]))
}


# defining biases
biases = {

    'h1' : tf.Variable(tf.truncated_normal([n_hidden_1])),
    'h2' : tf.Variable(tf.truncated_normal([n_hidden_2])),
    'h3' : tf.Variable(tf.truncated_normal([n_hidden_3])),
    'h4' : tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
}


# initializing all the global variable
init = tf.global_variables_initializer()


# prediction function.
y = model(x,weights,biases)


# defining cost function
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))

# training with optimixation
training_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_function)


# running using session

sess = tf.Session()
sess.run(init)


for i in range(epochs):
    sess.run(training_step,feed_dict={x:x_train,y_:y_train})
    cost = sess.run(cost_function,feed_dict={x:X,y_:Y})

    pred_y = sess.run(y,feed_dict={x:x_test})
    mse = tf.reduce_mean(tf.square(pred_y - y_test))
    mse_new = sess.run(mse)

    print("cost :{},epoch :{},MSE :{}".format(cost,i,mse_new))






