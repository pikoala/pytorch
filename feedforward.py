import torch


class Feedforward(torch.nn.Module):
    '''
    # Base class for all neural network modules.
    '''

    def __init__(self, input_size, output_size):
        '''
        # we initialize self but also input size and hidden size
        '''
        # self should also subclass of feedforward and we initialize
        super(Feedforward, self).__init__()
        # declare input_size and hidden_size
        self.input_size = input_size
        self.hidden_size_1 = 10
        self.hidden_size_2 = 9
        self.hidden_size_3 = 8
        self.hidden_size_4 = 7
        self.output_size = output_size
        # declare functions
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = torch.nn.Linear(self.input_size_2, self.hidden_size_3)
        self.fc4 = torch.nn.Linear(self.input_size_3, self.hidden_size_4)
        self.fc5 = torch.nn.Linear(self.hidden_size_4, self.output_size)
        self.sigmoid = torch. nn.Sigmoid()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # the input x is passed to the hidden layer (in-to-hidden)
        hidden_1 = self.fc1(x)
        # sigmoid is applied on the output of the hidden
        sigmoid_1 = self.sigmoid(hidden_1)
        # the input x is passed to the hidden layer (in-to-hidden)
        hidden_2 = self.fc2(sigmoid_1)
        # sigmoid is applied on the output of the hidden
        sigmoid_2 = self.sigmoid(hidden_2)
        # the input x is passed to the hidden layer (in-to-hidden)
        hidden_3 = self.fc3(sigmoid_2)
        # sigmoid is applied on the output of the hidden
        sigmoid_3 = self.sigmoid(hidden_3)
        # the input x is passed to the hidden layer (in-to-hidden)
        hidden_4 = self.fc4(sigmoid_3)
        # sigmoid is applied on the output of the hidden
        sigmoid_4 = self.sigmoid(hidden_4)
        # relu is passed to the ouptut layer (hidden-to-output)
        output = self.fc5(sigmoid_4)
        # softmax is applied to the output
        return self.softmax(output)
