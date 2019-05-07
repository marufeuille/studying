import gzip
import csv
import time
import switchCupy
xp = switchCupy.xp_factory()
from Network import TwoLayerNet

start = time.time()
with gzip.open("mnist/train-labels-idx1-ubyte.gz", "rb") as file:
    label_data = file.read()
magic = int.from_bytes(label_data[0:4], byteorder='big')
num_of_data = int.from_bytes(label_data[4:8], byteorder='big')
offset = 8
label = [int(s) for s in label_data[offset:]]
t_train = xp.identity(10)[label]
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
x_train = xp.array(x_train_tmp, dtype=xp.uint16)
print ("shape of train-images-idx3-ubyte: {}".format(x_train.shape))


with gzip.open("mnist/t10k-labels-idx1-ubyte.gz", "rb") as file:
    label_data = file.read()
magic = int.from_bytes(label_data[0:4], byteorder='big')
num_of_data = int.from_bytes(label_data[4:8], byteorder='big')
offset = 8
label = [int(s) for s in label_data[offset:]]
t_test = xp.identity(10)[label]
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
x_test = xp.array(x_test_tmp, dtype=xp.uint16)
print ("shape of t10k-images-idx3-ubyte: {}".format(x_test.shape))

num_of_showimg = 9885
print("label of {} : {}".format(num_of_showimg, t_train[num_of_showimg]))

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iter_num):
    batch_mask = xp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

