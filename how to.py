import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# (3) Automatic differentiation
x = torch.ones(3)  # input tensor
y = torch.zeros(4)  # expected output
w = torch.randn(3, 4, requires_grad=True)
b = torch.randn(4, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward()

print(w.grad)
print(b.grad)

z_det = z.detach()
print(z_det.requires_grad)


print('------------------')

# (2) tensors conversion and where it is stored
I = [[1, 0], [0, 1]]
np_array = np.array(I)

tsr = torch.tensor(I).to('mps')     # directly from array
tsr2 = torch.from_numpy(np_array)   # convert from numpy array

print(f"Params {tsr.device}, {tsr.shape}, {tsr.dtype}")

# (1) Data loader
training_data = datasets.FashionMNIST(
    root="FashionMNIST",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="FashionMNIST",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
# plt.show()