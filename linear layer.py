import torch

# linear layers

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('Layer params:')
for param in tinymodel.linear1.parameters():
    print(param)

# linear layer
fully_connected_layer = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
y = fully_connected_layer(x)

print(y)
print('-------params---------')
for param in fully_connected_layer.parameters():
    print(param)