
# Create the model
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#输入节点数
in_units = 784
#隐含层的输出节点数
h1_units = 300
#w1，b1是隐层的权重和偏置  赋值为0 权重初始化为截断的正态分布 标准差为0.1
#因为模型使用的激活函数是ReLU，所以需要使用正态分布给参数加一些噪声，
#来打破完全对称并且避免0梯度，有时候还需要给偏置一些小的非0值来避免死亡神经元
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
#输出层的softmax 直接将权重和偏置都初始化为0即可
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))


x = tf.placeholder(tf.float32, [None, in_units])
#在训练时keep_prob小于1，在预测时等于1
keep_prob = tf.placeholder(tf.float32)

#隐层，hidden1，一层
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# Train
tf.global_variables_initializer().run()
for i in range(3000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))