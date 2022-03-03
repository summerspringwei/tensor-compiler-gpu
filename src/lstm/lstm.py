import math
import numpy as np
from numpy.lib.histograms import _hist_bin_scott

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def test_sigmoid():
    print(sigmoid(3.0))
    print(sigmoid(sigmoid(3.0)))
    c_state = 1*sigmoid(513+1) + sigmoid(513)*math.tanh(513)
    print(c_state)
    h_state = math.tanh(c_state) * sigmoid(513)
    print(h_state)


num_hidden=256


def generate_one_lstm_cell_inputs_data():
    input = np.ones((num_hidden))
    c_state = np.ones((num_hidden))
    h_state = np.ones((num_hidden))
    bias = np.ones((num_hidden))
    np.save("data/input.npy", input)
    np.save("data/c_state.npy", c_state)
    np.save("data/h_state.npy", h_state)
    np.save("data/bias.npy", bias)

    for i in range(4):
        w = np.ones((num_hidden * num_hidden))
        np.save("data/W_{}.npy".format(i), w)
        u = np.ones((num_hidden * num_hidden))
        np.save("data/U_{}.npy".format(i), u)


def generate_random_lstm_cell_inputs_data():
    input = np.random.rand(num_hidden).astype(np.float32)
    c_state = np.random.rand(num_hidden).astype(np.float32)
    h_state = np.random.rand(num_hidden).astype(np.float32)
    bias = np.random.rand(num_hidden).astype(np.float32)
    
    np.save("data/input.npy", input)
    np.save("data/c_state.npy", c_state)
    np.save("data/h_state.npy", h_state)
    np.save("data/bias.npy", bias)

    for i in range(4):
        w = np.ones((num_hidden * num_hidden), dtype=np.float32)
        np.save("data/W_{}.npy".format(i), w)
        u = np.ones((num_hidden * num_hidden), dtype=np.float32)
        np.save("data/U_{}.npy".format(i), u)


import tensorflow as tf

class LSTMCell(object):
    W = []
    U = []
    b = []

    def __init__(self, hidden_size, scope):
        self.W = []
        self.U = []
        self.b = []
        self.num_unit = hidden_size
        for i in range(4):
            W = np.load("data/W_{}.npy".format(i))
            U = np.load("data/U_{}.npy".format(i))
            b = np.load("data/bias.npy")
            W = np.reshape(W, (num_hidden, num_hidden))
            U = np.reshape(W, (num_hidden, num_hidden))
            self.W.append(W)
            self.U.append(U)
            self.b.append(b)

    def call(self, inputs, state):
        c, h = state
        res = []
        for i in range(4):
            res.append(tf.matmul(
                inputs, self.W[i]) + tf.matmul(h, self.U[i]) + self.b[i])
        i, j, f, o = (res[0], res[1], res[2], res[3])
        new_c = (c * tf.sigmoid(f + 1.0) +
                tf.sigmoid(i) * tf.tanh(j))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
        new_state = (new_c, new_h)
        return new_h, new_state

def run_lstm_cell():
    input = np.load("data/input.npy")
    c_state = np.load("data/c_state.npy")
    h_state = np.load("data/h_state.npy")
    print("input")
    print(input)
    print("c_state")
    print(c_state)
    print("h_state")
    print(h_state)

    input=np.reshape(input, (1, num_hidden))
    c_state=np.reshape(c_state, (1, num_hidden))
    h_state=np.reshape(h_state, (1, num_hidden))
    lstm_cell = LSTMCell(256, "LSTM")
    c_state, state = lstm_cell.call(input, (c_state, h_state))

    print("c_state")
    print(state[0])
    print("h_state")
    print(state[1])


def call(self, inputs, state):
    c, h = state
    res = []
    for i in range(4):
        res.append(math_ops.matmul(
            inputs, self.W[i]) + math_ops.matmul(h, self.U[i]) + self.b[i])
    i, j, f, o = (res[0], res[1], res[2], res[3])
    new_c = (c * math_ops.sigmoid(f + 1.0) +
            math_ops.sigmoid(i) * math_ops.tanh(j))
    new_h = math_ops.tanh(new_c) * math_ops.sigmoid(o)
    new_state = (new_c, new_h)
    return new_h, new_state


if __name__=="__main__":
    # generate_one_lstm_cell_inputs_data()
    # generate_random_lstm_cell_inputs_data()
    # test_sigmoid()
    run_lstm_cell()
