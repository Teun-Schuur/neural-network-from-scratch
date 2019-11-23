from NeuralNetwork import *


class NeuralNetwork():
    def __init__(self, topology, learningRate, momentum):
        Neuron.eta = learningRate
        Neuron.alpha = momentum
        self.net = Network(topology)
        self.inputs = []
        self.outputs = []
        self.topology = topology

    def setInputOutput(self, input, output):
        self.outputs = output
        self.inputs = input

    def train(self, error, time, update):
        __times = 0
        while True:
            __times += 1
            err = 0
            for i in range(len(self.inputs)):
                self.net.setInput(self.inputs[i])
                self.net.feedForword()
                err = err + self.net.getError(self.outputs[i])
                self.net.backPropagate(self.outputs[i])
            if __times % update == 0:
                print("cycels: ", __times, "     error: ", round(err, 5), "       eta:", round(Neuron.eta, 4), "       alpha:", round(Neuron.alpha, 4))
            if err < error or __times == time:
                break

    def giveInput(self, data, outputs, natural=False):
        self.net.setInput(data)
        self.net.feedForword()
        results = []
        for i in range(outputs):
            results.append(float(self.net.getThResults(natural)[i]))
        return results
