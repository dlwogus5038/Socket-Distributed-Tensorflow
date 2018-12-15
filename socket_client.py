import socket,select,threading,sys,os;
import input_data
import tensorflow as tf
import time
 
server_ID=socket.gethostname()
server_addr=(server_ID,5964)

tf.set_random_seed(777)  # reproducibility
 
def conn():
    s=socket.socket()
    s.connect(server_addr)
    return s

def run_tensor_client(data,s):
    str = data.split('/')
    ps_hosts = (str[1]).split(',')
    worker_hosts = (str[2]).split(',')
    task_index = int(str[3])

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name="worker",
                             task_index=task_index)
    print("Cluster job: %s, task_index: %d, target: %s" % ("worker", task_index, server.target))
    if "worker" == "worker":
        
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            # Build model ...
            mnist = input_data.read_data_sets("data", one_hot=True)

            # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
            keep_prob = tf.placeholder(tf.float32)
            
           # Create the model
            x = tf.placeholder(tf.float32, [None, 784])

            # weights & bias for nn layers
            W1 = tf.get_variable("W1", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([512]))
            L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
            L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

            W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([512]))
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
            L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

            W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([512]))
            L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
            L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

            W4 = tf.get_variable("W4", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([10]))

            y = tf.matmul(L3, W4) + b4

            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 10])
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

            global_step = tf.Variable(0)

            train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()


        # Create a "Supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                 logdir="/opt/tensor",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver = saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        sess = sv.prepare_or_wait_for_session(server.target)

        # Start queue runners for the input pipelines (if ang).
        sv.start_queue_runners(sess)

        # Loop until the supervisor shuts down (or 1000 steps have completed).
        step = 0
        while not sv.should_stop() and step < 1000:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, _ = sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.7})
            print("Step %d in task %d" % (step, task_index))
            step = step + 1
        print("done.")
        print("Accuracy: %f" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                             y_: mnist.test.labels, keep_prob: 1}))
        data = "END!!"
        try:
            s.send(data.encode())
        except Exception as e:
            print("Send Error")
            s.close()
            return

        if task_index != 0:
            return
        elif task_index == 0:
            while True:
                data=s.recv(1024).decode()
                print(data)
                if data == "0":
                    time.sleep(3)
                    break
                elif data == '':
                    time.sleep(3)
                    break
        s.close()


def lis(s):
    my=[s]
    data = ""
    while True:
        r,w,e=select.select(my,[],[])
        if s in r:
            try:
                data = s.recv(1024).decode()
                if data == '':
                    print("server closed")
                    s.close()
                    break
                elif data[0:7] == "START!!":
                    print("Distributed Tensorflow Client is running..")
                    time.sleep(10)
                    run_tensor_client(data,s)
                    break
                print (data)
            except socket.error:
                print ('socket error')
                s.close()
                break
            

if __name__ == "__main__":
    ss=conn()
    lis(ss)
    print("Succeed!")
