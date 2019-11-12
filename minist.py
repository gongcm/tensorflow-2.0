import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 



#def loss():
    
# 单层手写识别

# y = w * x + b
mnist = tf.keras.datasets.mnist 
#print(type(mnist))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("tensorflow version : ",tf.version)
print("type(shape) ",type(x_train.shape),x_train.T.shape,len(x_train))
w = np.empty(y_train.shape[0],dtype=x_train.dtype)
b = np.random.random(y_train.shape)
print("x data ",x_train.shape," y data ",y_train.shape,"w shape ",w.shape,"b shape ",b.shape)

print("x.shape[0] ",x_train.shape[0])
print("x.shape[1] ",x_train.shape[1])
print("x.shape[0:1] ",x_train.shape[0:1])
print("x.shape[0:1] ",x_train.shape[1:len(x_train.shape)-1])
print("y labels ",y_train,"\n y shape ",y_train.shape)
print(b)

fig,ax= plt.subplots(nrows=10,ncols=10,sharex=True,sharey=True)
print("type(ax) ",type(ax))
ax = ax.flatten()

for i in range(100):
    img = x_train[i].reshape(28,28)
    ax[i].imshow(img)

fig.show()
plt.savefig("minst.png",dpi=400)

y = tf.matmul(x_train.T,w) + b 

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_train,y))

y_train_a = tf.

# (2,3) * (3,2) = (2,2)
#y = tf.matmul()

#print()
