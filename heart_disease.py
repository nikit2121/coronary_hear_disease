import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import pandas as pd 

hidden_1_neurons = 100
hidden_2_neurons = 100
hidden_3_neurons = 100
input_size = 4
class_label = 1
batch_size = 50

data = np.genfromtxt('/home/nikit/Desktop/coronary_heart_disease/saheart_1_withheader.csv',delimiter=',')
data = np.delete(data,0,axis=0)
target = data[:,0]
imp_features = [1,2,3,9]
data = data[:,imp_features]
data = data/data.max(axis=0)
#lb = LabelBinarizer()


target_label= np.zeros((len(target),1),dtype=np.float32)
for i in range(0,462):
    if target[i]==1:
        target_label[i]=1
    else:
        target_label[i]=0
#target = lb.fit_transform(target).astype(np.float32)

#data = data[:,1:]
train_data = data[:400,:]
test_data = data[400:,:]
train_label = target_label[:400,:]
test_label = target_label[400:,:]

X = tf.placeholder(tf.float32,[None,input_size])
y = tf.placeholder(tf.float32,[None,1])

weights = {'w1': tf.Variable(tf.truncated_normal([input_size,hidden_1_neurons])),
           'w2': tf.Variable(tf.truncated_normal([hidden_1_neurons,hidden_2_neurons])),
           'w3': tf.Variable(tf.truncated_normal([hidden_2_neurons,hidden_3_neurons])),
           'out':tf.Variable(tf.truncated_normal([hidden_3_neurons,class_label]))}

biases = {'b1': tf.Variable(tf.truncated_normal([hidden_1_neurons])),
           'b2': tf.Variable(tf.truncated_normal([hidden_2_neurons])),
           'b3': tf.Variable(tf.truncated_normal([hidden_3_neurons])),
           'b4':tf.Variable(tf.truncated_normal([class_label]))}

def multilayer_perceptron(X):
    #X = tf.nn.l2_normalize(X,dim=0)
    layer_1 = tf.nn.sigmoid((tf.matmul(X,weights['w1'])+biases['b1']))
    layer_2 = tf.nn.sigmoid((tf.matmul(layer_1,weights['w2'])+biases['b2']))
    layer_3 = tf.nn.sigmoid((tf.matmul(layer_2,weights['w3'])+biases['b3']))
    out_layer = tf.nn.sigmoid((tf.matmul(layer_3,weights['out'])+biases['b4']))
    return out_layer

prediction = multilayer_perceptron(X)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
cost = tf.reduce_sum(tf.square(y-prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
init = tf.global_variables_initializer()
epoch_size = 20
count=np.zeros((epoch_size,1))
current_cost=np.zeros((epoch_size,1))
iterator=np.zeros((epoch_size,1))
with tf.Session() as session:
    session.run(init)
    for epochs in range(0,epoch_size):
        n= 0
        for step in range(0,400/batch_size):
            offset = step*batch_size
            batch_x = train_data[offset:(offset+batch_size),:]
            batch_y = train_label[offset:(offset+batch_size),:]
            _,c,p = session.run([optimizer,cost,prediction],feed_dict={X:batch_x,y:batch_y})
            current_cost[epochs,0] = c
            iterator[epochs,0]=epochs+1
        print("epoch ", '%02d' %(epochs), "Cost:" '%.2f'%(c))

        pred = session.run(prediction,feed_dict={X:test_data,y:test_label})
        pred = np.asarray(pred)
        for i in range(0,len(pred)):
            if pred[i]<0.5:
                pred[i]=0
            else:
                pred[i]=1
        for i in range(0,len(pred)):
            if pred[i]==test_label[i]:
                n=n+1
        count[epochs,0]=n
        print("optimized!")
    accuracy = n/i
    print(accuracy)
    plt.plot(iterator,current_cost)
    plt.plot(iterator,count)
    plt.legend(['cost','accuracy'])
    plt.title('COST AND ACCURACY after each epoch')
    plt.show()
    session.close()

print('hello')

