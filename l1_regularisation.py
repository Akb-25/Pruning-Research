import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt

x_train=torch.randn(1,1)
y_train=torch.randn(1,1)

dataset=TensorDataset(x_train,y_train)
print("------------------------------------")
print("TENSOR DATASET")
print(dataset)


batch_size=1
dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

model=nn.Sequential(
    nn.Linear(1,5),
    nn.ReLU(),
    nn.Linear(5,1)
)
criterion = nn.MSELoss()

optimizer=optim.SGD(model.parameters(),lr=0.01,weight_decay=0.01)
count=0
for inputs,targets in dataloader:
    print("--------------------------------")
    print("INPUTS")
    print(inputs)

    print("--------------------------------")
    print("LENGTH OF INPUT")
    print(len(inputs))

    optimizer.zero_grad()
    outputs=model(inputs)
    print("--------------------------------")
    print("OUTPUTS")
    print(outputs)
    loss=criterion(outputs,targets)
    print("--------------------------------")
    print("LOSS")
    print(loss)
    l1_regularisation=0
    for param in model.parameters():
        print("--------------------------------")
        print("PARAM")
        print(param)
        l1_regularisation+=torch.norm(param,p=1)
        print("--------------------------------")
        print("L1 REGULARISATION")
        print(l1_regularisation)
        count+=1
    lambda_val=0.01
    
    loss+=lambda_val*l1_regularisation
    print("--------------------------------")
    print("LOSS - L1 REGULARISATION")
    print(loss)
    loss.backward()
    optimizer.step()