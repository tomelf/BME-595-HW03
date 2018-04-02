# BME-595 Assignment 03

1. Create a Python script NeuralNetwork.py that contains two local empty dictionaries Theta and dE_dTheta and the five functions described in the API.
2. Implement back-propagation with a Mean Square Error loss function.
3. Update the matrices Theta with updateParams(eta) based on the learning rate eta and the gradient of the error with respect of the parameters dE_dTheta.
4. Train the AND, OR, NOT and XOR networks using the NeuralNetwork API (calling forward(), backward() and updateParams() in a cycle) on a hand-crafted data set (Hint: use Python’s and, or, not and combination of these in order to build your data on the fly). Compare the the Thetas with what you set manually in HW02. One may learn the new Thetas with the train() function.


### API (NeuralNetwork)
- build: Take the numbers of layers as input, randomly initialize a series of theta layers (including bias layer)
- getLayer: Take one index of theta layers as input, return the specific theta layer
- forward: Take a 1D tensor [DoubleTensor size k] or a 2D tensor [DoubleTensor size k x n] as input (where k is the number of input features, n is the number of input vectors), forward the input to all layers and return the output. Record a, a_hat
- backward: Take a 1D tensor [DoubleTensor size k] or a 2D tensor [DoubleTensor size k x n] as input (where k is the number of output layer size, n is the number of output vectors), back propagate the input to all layers and return the output. Record delta, dE_dTheta
- updateParams: Update eta, and perform theta update: Theta = Theta - self.eta * dE_dTheta

### API (logicGates)

- Each class extends the class NeuralNetwork
- Constructor: For AND, OR, XOR, initialize NeuralNetwork(2,2,1). For NOT, initialize NeuralNetwork(1,2,1)
- Call: For AND, OR, XOR, perform forward(x, y). For NOT, perform forward(x). Return the output
- forward: Convert boolean to (0,1) as the input of forward() function. Return True if the output > threshold(0.5) else False.
- train: For AND, OR, XOR, perform forward(x, y). For NOT, perform forward(x). Then all classes calculate the expected output value as the input of backward() function, and then call updateParams() to update Thetas

### Observations

1. When setting eta=0.2, threshold=0.5, all four logic gates converge in 2000-3000 training cycles. Each training cycle takes the following example:
- AND: (TRUE, TRUE)->TRUE, (TRUE, FALSE)->FALSE, (FALSE, TRUE)->FALSE, (FALSE, FALSE)->FALSE
- OR: (TRUE, TRUE)->TRUE, (TRUE, FALSE)->TRUE, (FALSE, TRUE)->TRUE, (FALSE, FALSE)->FALSE
- NOT: (TRUE)->FALSE, (FALSE)->TRUE
- XOR: (TRUE, TRUE)->FALSE, (TRUE, FALSE)->TRUE, (FALSE, TRUE)->TRUE, (FALSE, FALSE)->FALSE

2. Comparing to assignment 2, it is much easier to add more levels of layer and build large layers
