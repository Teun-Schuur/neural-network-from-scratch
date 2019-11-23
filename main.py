from Training import *
import numpy as np

inputs = [[1, 0], [0, 2], [3, 1], [4, 2], [1, 4],
          [5, 1], [0, 0], [6, 3], [2, 2], [5, 1],
          [1, 2], [6, 2], [3, 6], [2, 2], [3, 5],
          [6, 5], [7, 2], [3, 7], [9, 9], [10, 5],
          [0, 9], [8, 6], [4, 7], [9, 2], [3, 8],
          [7, 6], [2, 9], [5, 6], [9, 8], [7, 9]]

outputs = [[1], [2], [4], [6], [5],
           [6], [0], [9], [4], [6],
           [3], [8], [9], [4], [8],
           [11], [9], [10], [18], [15],
           [9], [14], [11], [11], [11],
           [13], [11], [11], [17], [16]]

i = 0
for inp in inputs:
    inputs[i][0] = inp[0] / 10
    inputs[i][1] = inp[1] / 10
    i += 1
i = 0
for out in outputs:
    outputs[i][0] = out[0] / 20
    i += 1

topology = [2, 3, 1]

learningRate = 0.5
momentum = 0.09
error = 0.08


def main():
    NN = NeuralNetwork(topology, learningRate, momentum)
    NN.setInputOutput(inputs, outputs)
    NN.train(error, 100000, 500)

    while True:
        data = [int(input("type 1st input :"))/10, int(input("type 2nd input :"))/10]
        final = float(NN.giveInput(data, 1)[0])*20
        print(str(np.round(final, 0))[:-2])


if __name__ == '__main__':
    main()
