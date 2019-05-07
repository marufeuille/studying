import gzip
import cupy as cp
import numpy as np
import csv
import time

start = time.time()
with gzip.open("mnist/train-labels-idx1-ubyte.gz", "rb") as file:
    label_data = file.read()
magic = int.from_bytes(label_data[0:4], byteorder='big')
num_of_data = int.from_bytes(label_data[4:8], byteorder='big')
offset = 8
label = [int(s) for s in label_data[offset:]]
t_train = cp.identity(10)[label]
# too slow
# for i in range(0, num_of_data):
#   t_train [i] = cp.zeros(10,dtype="int8")
#   t_train [i][label_data[offset+i]] = 1
print ("shape of train-labels-idx1-ubyte: {}".format(t_train.shape))

with gzip.open("mnist/train-images-idx3-ubyte.gz", "rb") as file:
    dataset = file.read()
magic = int.from_bytes(dataset[0:4], byteorder='big')
num_of_data = int.from_bytes(dataset[4:8], byteorder='big')
num_of_rows = int.from_bytes(dataset[8:12], byteorder='big')
num_of_cols = int.from_bytes(dataset[12:16], byteorder='big')
data_size = num_of_rows * num_of_cols
x_train_tmp = []
offset = 16
dataset_tmp = [int(s) for s in dataset[offset:offset+data_size*num_of_data]]
for i in range(0, num_of_data):
    x_train_tmp.append(dataset_tmp[i*data_size:(i+1)*data_size])
#     for j in range(0, data_size):
#       x_train[i][j] = dataset[offset+i*data_size+j]
x_train = cp.array(x_train_tmp, dtype=cp.uint16)
print ("shape of train-images-idx3-ubyte: {}".format(x_train.shape))


with gzip.open("mnist/t10k-labels-idx1-ubyte.gz", "rb") as file:
    label_data = file.read()
magic = int.from_bytes(label_data[0:4], byteorder='big')
num_of_data = int.from_bytes(label_data[4:8], byteorder='big')
offset = 8
label = [int(s) for s in label_data[offset:]]
t_test = cp.identity(10)[label]
# too slow
# for i in range(0, num_of_data):
#   t_test [i] = cp.zeros(10,dtype="uint8")
#   t_test [i][label_data[offset+i]] = 1

print ("shape of t10k-labels-idx1-ubyte: {}".format(t_test.shape))


with gzip.open("mnist/t10k-images-idx3-ubyte.gz", "rb") as file:
    dataset = file.read()
magic = int.from_bytes(dataset[0:4], byteorder='big')
num_of_data = int.from_bytes(dataset[4:8], byteorder='big')
num_of_rows = int.from_bytes(dataset[8:12], byteorder='big')
num_of_cols = int.from_bytes(dataset[12:16], byteorder='big')
data_size = num_of_rows * num_of_cols
x_test_tmp = []
offset = 16
dataset_tmp = [int(s) for s in dataset[offset:offset+data_size*num_of_data]]
for i in range(0, num_of_data):
    x_test_tmp.append(dataset_tmp[i*data_size:(i+1)*data_size])
x_test = cp.array(x_test_tmp, dtype=cp.uint16)

# x_test = cp.ndarray((num_of_data, data_size), dtype='uint8')
# offset = 16
# for i in range(0, num_of_data):
#     for j in range(0, data_size):
#       x_test[i][j] = dataset[offset+i*data_size+j]

print ("shape of t10k-images-idx3-ubyte: {}".format(x_test.shape))

num_of_showimg = 9885
print("label of {} : {}".format(num_of_showimg, t_train[num_of_showimg]))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - cp.max(x, axis=0)
        y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
        return y.T 

    x = x - cp.max(x)
    return cp.exp(x) / cp.sum(cp.exp(x))


def sigmoid(x):
    return 1/(1+cp.exp(-x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = cp.zeros_like(x)
    it = cp.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = float(tmp_val) - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        #print("01 -> fxh1 = {}".format(fxh1))
        #print("01 -> fxh2 = {}".format(fxh2))

        x[idx] = tmp_val # 値を元に戻す
        
        it.iternext()

    return grad


input_size=784
hidden_size=50
output_size=10
weight_init_std=0.01

class Params:
    def __init__(self):
        self.params = {}
        self.params['W1'] = weight_init_std * cp.random.randn(input_size, hidden_size)
        self.params['b1'] = cp.zeros(hidden_size)
        self.params['W2'] = weight_init_std * cp.random.randn(hidden_size, output_size)
        self.params['b2'] = cp.zeros(output_size)

p = Params()


iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


def predict(x):
    W1, W2 = p.params['W1'], p.params['W2']
    b1, b2 = p.params['b1'], p.params['b2']
    
    a1 = cp.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = cp.dot(z1, W2) + b2
    y = softmax(a2)
        
    return y


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def gradient(x, t):
    W1, W2 = p.params['W1'], p.params['W2']
    b1, b2 = p.params['b1'], p.params['b2']
    grads = {}

    batch_num = x.shape[0]

    # forward
    a1 = cp.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = cp.dot(z1, W2) + b2
    y = softmax(a2)

    # backward
    dy = (y - t) / batch_num
    grads['W2'] = cp.dot(z1.T, dy)
    grads['b2'] = cp.sum(dy, axis=0)

    dz1 = cp.dot(dy, W2.T)
    da1 = sigmoid_grad(a1) * dz1
    grads['W1'] = cp.dot(x.T, da1)
    grads['b1'] = cp.sum(da1, axis=0)

    return grads


results = []
results.append([])
results[0].append("train_loss")
results[0].append("train_acc")
results[0].append("test_acc")

iter_per_epoch = int(max(train_size / batch_size, 1))

start_analysis = time.time()
for i in range(iters_num):

    batch_mask = cp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    loss_W = lambda W: loss(x_batch, t_batch)
    
    grads = {}
    grads['W1'] = numerical_gradient(loss_W, p.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, p.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, p.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, p.params['b2'])
    
    
    #grads = gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        p.params[key] -= learning_rate * grads[key]

    y = predict(x_batch)
    loss_val = cross_entropy_error(y, t_batch)
    
    if i % iter_per_epoch == 0:
        
        y = predict(x_train)
        y = cp.argmax(y, axis=1)
        t = cp.argmax(t_train, axis=1)
        train_acc = cp.sum(y == t) / float(x_train.shape[0])
        
        y = predict(x_test)
        y = cp.argmax(y, axis=1)
        t = cp.argmax(t_test, axis=1)
        test_acc = cp.sum(y == t) / float(x_test.shape[0])
        
        results.append([loss_val, train_acc, test_acc])
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

analysis_elapsed = time.time() - start_analysis

with open("./train_graph.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)

all_elapsed = time.time() - start

print("Total time elapsed: {} [sec]".format(all_elapsed))
print("Analysis time elapsed: {} [sec]".format(analysis_elapsed))
