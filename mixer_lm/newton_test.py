import torch

def newton_iteration(model, train_batch, target, loss_constant=0.9):
        for i in range(10):
                output = model(train_batch)
                loss = mse(output, target) - loss_constant # subtract suspected irreducible loss so root exists
                print (f"Starting loss: {(loss)}")
                loss.backward()
                loss_term = torch.pinverse(model.weight.grad) * loss
                model.weight = torch.nn.Parameter(model.weight - loss_term.T)
                with torch.no_grad(): 
                        output = model(train_batch)
                        loss = mse(output, target) - loss_constant
                        print (f"Ending loss: {loss} \n")
        return 

def normal_solve(model):
        X = torch.stack([torch.tensor([30.,10.]),torch.tensor([-3., 5.])], dim=0)
        target = torch.stack([torch.tensor([0., 0.])], dim=0)
        output = model(X)
        loss = mse(output, target) # learns an algebraic kernel
        print (f"Starting loss: {(loss)}")
        print ((torch.inverse(X.T @ X) @ X.T).shape, target.shape)
        beta_hat = torch.inverse(X.T @ X) @ X.T * target
        with torch.no_grad():
                model.weight = torch.nn.Parameter(beta_hat)
                output = model(X)
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
model = torch.nn.Linear(200, 1, bias=False)
# normal_solve(model)
# train_batch = torch.stack([torch.tensor([5., -1., 1., 2.]),torch.tensor([-3., 5., -2., 0.])], dim=0)
# target = torch.stack([torch.tensor([7.]), torch.tensor([1.])], dim=0)
train_batch = torch.randn(1000, 200)
target = torch.randn(1000, 1)
newton_iteration(model, train_batch, target)
