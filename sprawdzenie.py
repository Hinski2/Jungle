import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

X_train = torch.FloatTensor([0., 1., 2.])
X_train = X_train.to(device)
print(X_train.is_cuda)

