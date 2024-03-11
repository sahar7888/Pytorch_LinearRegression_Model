


"""Ref:PyTorch for Deep Learning Bootcamp
 """


import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from plotting import PLOT
from LinearRegressionModel import LinearRegressionModel

""" Data Preparation"""
# Defining the coefficients
weight = 0.7
bias = 0.3
# setting up the data range
start = 0
end = 1
step = 0.02
# Creating a linear data
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias

""" Split Data"""
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

""" Plot train vs test data"""
#
plot1 = PLOT(X_train, Y_train, X_test, Y_test)

plot1.plot_predictions()

""" Create the model"""
torch.manual_seed(42)
model_0 = LinearRegressionModel()
# print(model_0.state_dict())

""" making a prediction using the test data"""
with torch.inference_mode():
    y_preds = model_0(X_test)

# print(y_preds)

""" Plot the predicted data"""

# plot1.plot_predictions(predictions=y_preds)

""" set up the loss function"""

loss_fn = nn.L1Loss()

"""set up the optimizer"""
optimizer = torch.optim.SGD(model_0.parameters(),
                            lr=0.01)

""" Creating the training loop"""

epochs = 150
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    #set the model to training mode:
    model_0.train()

    #Forward pass:
    y_pred = model_0(X_train)
    # print("working")

    #calculate the loss
    loss = loss_fn(y_pred, Y_train)

    #Set the optimizer to 0 grad
    optimizer.zero_grad()

    # Perform BP
    loss.backward()

    # Preform Gradient Descent
    optimizer.step()


    ## Testing
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)

    #Calculate the loss
    test_loss = loss_fn(test_pred, Y_test)

    #Print results
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}| Loss: {loss} | Test Loss: {test_loss}")

        print(model_0.state_dict())

        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)


""" Plot the training process"""

plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label ='Train loss')
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Loss curve")
plt.ylabel(epochs)
plt.legend()
plt.show()

"""The prediction plot"""

plot1.plot_predictions(predictions=test_pred)

""" Saving the model"""
#Creating the save path
MODEL_PATH ="models/EX_1.pth"
torch.save(obj = model_0.state_dict(), f="models/EX_1.pth")

""" Loading the model """

#load the model
loaded_model_0 = LinearRegressionModel()
print(loaded_model_0.state_dict())
# Update the new instance:
loaded_model_0.load_state_dict(torch.load(f=MODEL_PATH))

""" making a prediction"""

loaded_model_0.eval()
with torch.inference_mode():
  loaded_model_preds = loaded_model_0(X_test)


#Compare loaded model preds with original model
print(test_pred == loaded_model_preds)