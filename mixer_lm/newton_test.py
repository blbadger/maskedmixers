import torch

def newton_iterations(model, train_batch, target, loss_constant=0.0):
	mse = torch.nn.MSELoss()
	for i in range(10):
		loss_terms = []
		output = model(train_batch)
		loss = mse(output, target) - loss_constant # subtract suspected irreducible loss so root exists
		print (f"Starting loss: {(loss)}")
		loss.backward(retain_graph=True)
		loss_term = (torch.pinverse(model.weight.grad) * loss).T
		model.weight = torch.nn.Parameter(model.weight - loss_term)
		with torch.no_grad():
			output = model(train_batch)
			loss = mse(output, target) - loss_constant
			print (f"Ending loss: {torch.mean(loss)} \n")
		model.zero_grad()
	return

def newton_iterations_components(model, train_batch, target, loss_constant=0.0):
	mse = torch.nn.MSELoss(reduction='none')
	for i in range(10):
		loss_terms = []
		output = model(train_batch)
		loss = mse(output, target) - loss_constant # subtract suspected irreducible loss so root exists
		for j in range(3):
			loss[0][j].backward(retain_graph=True)
			loss_term = torch.pinverse(model.weight.grad) * loss[0][j]
			loss_terms.append(loss_term)
			model.zero_grad()

		for loss_term in loss_terms:
			model.weight = torch.nn.Parameter(model.weight - loss_term.T)
		print (f"Loss: {torch.sum(loss)}")
	return

def newton_iterations_recalculated(model, train_batch, target, loss_constant=0.0):
	for i in range(10):
		loss_terms = []
		for j in range(3):
			output = model(train_batch)
			loss = mse(output, target) - loss_constant # subtract suspected irreducible loss so root exists
			print (f"Starting loss: {(loss)}")
			loss[0][j].backward()
			model.weight = torch.nn.Parameter(model.weight - (torch.pinverse(model.weight.grad) * loss[0][j]).T)
			model.zero_grad()
		
		with torch.no_grad():
			output = model(train_batch)
			loss = mse(output, target) - loss_constant
			print (f"Ending loss: {torch.mean(loss)} \n")
		model.zero_grad()
	return

def compute_loss(train_batch, target):
	output = model(train_batch)
	error = mse(output, target)
	return error

def newton_jacobian(model, train_batch, target):
	for i in range(10):
		output = torch.tensor(model(train_batch))
		loss = mse(output, target)
		# print (f"Starting loss: {(loss)}")
		jacobian = torch.autograd.functional.jacobian(compute_loss, (train_batch, target))[0]
		# print (jacobian.shape)
		loss_term = torch.pinverse(jacobian) @ loss
		print (loss_term.shape)
		output = torch.nn.Parameter(output - loss_term.T)
		# print (output)
		with torch.no_grad():
			# output = model(train_batch)
			loss = mse(output, target)
			# print (f"Ending loss: {loss} \n")
		model.zero_grad()
	return 

def normal_solve(model, X, target):
	output = model(X)
	loss = mse(output, target) # learns an algebraic kernel
	print (f"Starting loss: {torch.mean(loss)}")
	beta_hat = torch.pinverse(X) @ target
	with torch.no_grad():
		model.weight = torch.nn.Parameter(beta_hat.T)
		output = model(X)
		loss = mse(output, target)
		print (output)
		print (f"Ending loss: {torch.mean(loss)} \n")
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

mse = torch.nn.MSELoss(reduction='none')
mse_nored = torch.nn.MSELoss(reduction='none')
model = torch.nn.Linear(4, 3, bias=False)
# normal_solve(model)
train_batch = torch.stack([torch.tensor([5., -1., 1., 2.]), torch.tensor([3., 0., -3., -1.])], dim=0)
target = torch.stack([torch.tensor([7., 8., 0.]), torch.tensor([-1., -9., 0.])], dim=0)
# train_batch = torch.randn(1000, 100)
# target = torch.randn(1000, 100)
# newton_iterations(model, train_batch, target)
# newton_iterations_components(model, train_batch, target)
# newton_iterations_recalculated(model, train_batch, target)
normal_solve(model, train_batch, target)

