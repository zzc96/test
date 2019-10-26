import tensorflow as tf

#创建W和b两个占位符
W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")

#定义计算函数   y = Wx+b
def inference(X):
    return tf.matmul(X, W) + b

#定义损失函数，采用sum(x-y)^2损失函数作为损失函数
def loss(X, Y):
    Y_predicted = tf.transpose(inference(X)) # 计算实际值,并且进行转置
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

#定义数据，体重-年龄和血脂
def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]

    return tf.to_float(weight_age), tf.to_float(blood_fat_content)

#定义优化函数
def train(total_loss):
    learning_rate = 0.000001                                              #初始化学习速率
                                                                          # 使用GD作为优化函数
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

#模型评估
def evaluate(sess, X, Y):
    print(sess.run(inference([[50., 20.]]))) # ~ 303
    print(sess.run(inference([[50., 70.]]))) # ~ 256
    print(sess.run(inference([[90., 20.]]))) # ~ 303
    print(sess.run(inference([[90., 70.]]))) # ~ 256


# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
    tf.initialize_all_variables().run()                                   #初始化所有变量

    X, Y = inputs()                                                       #获得输入

    total_loss = loss(X, Y)                                               #定义损失
    train_op = train(total_loss)                                          #得到Op

    coord = tf.train.Coordinator()                                        #创建线程管理器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)        #定义线程

    # actual training loop
    training_steps = 10000                                                #训练10000次
    for step in range(training_steps):
        sess.run([train_op])                                              #开始训练
        if step % 1000 == 0:                                              #每一千次输出一次
            print("Epoch:", step, " loss: ", sess.run(total_loss))

    print("Final model W=", sess.run(W), "b=", sess.run(b))               #输出最终训练出来的参数
    evaluate(sess, X, Y)                                                  #进行模型评估

    coord.request_stop()                                                  #请求线程停止
    coord.join(threads)                                                   #回收线程
    sess.close()                                                          #关闭会话


