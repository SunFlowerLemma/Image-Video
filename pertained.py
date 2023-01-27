from torch import nn, optim
from torchvision import models

# the pretrained model
model_pt = models.resnet18(pretrained=True)

input_size, output_size = model_pt.fc.in_features, 2 # print to check
model_pt.fc = nn.Linear(input_size, output_size)

model_pt = model_pt.to('mps')
criterion = nn.CrossEntropyLoss()

# stochastic gradient descent
optimizer = optim.SGD(model_pt.parameters(), lr=0.001, momentum=0.9)

learning_rate = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
