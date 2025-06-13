import torch

def newton_iteration(model):
        train_batch = torch.stack([torch.tensor([100.,]),torch.tensor([5.])], dim=0)
        target = torch.stack([torch.tensor([0.])], dim=0)
        output = model(train_batch)
        loss = mse(output, target) # learns the identity function
        print (f"Starting loss: {(loss)}")
        torch.mean(loss).backward() # gradients propegated to params
        # print (model.weight.grad)
        mini = model.weight - torch.inverse(model.weight.grad) * loss
        with torch.no_grad(): 
                model.weight = torch.nn.Parameter(mini)
                output = model(train_batch)
                loss = mse(output, target)
                print (output)
                print (f"Ending loss: {loss} \n")
        return 

def grad_descent(model):
        train_batch = torch.stack([torch.tensor([0.,1.]),torch.tensor([2.,3.])], dim=0)
        target = torch.stack([torch.tensor([0.,0.]),torch.tensor([0.,0.])], dim=0)
        output = model(train_batch)
        loss = mse(output, target) # learns the identity function
        print (f"Starting loss: {(loss)}")
        loss.backward() # gradients propegated to params
        with torch.no_grad(): 
                model.weight -= 0.005 * model.weight.grad
                output = model(train_batch)
                loss = mse(output, target)
                print (output)
                print (f"Ending loss: {loss} \n")
        return 

mse = torch.nn.MSELoss()
model = torch.nn.Linear(1, 1, bias=False)
for i in range(3):
        newton_iteration(model)
