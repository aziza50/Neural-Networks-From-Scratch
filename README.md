# Neural Networks From Scratch

To uncover the inner workings behind neural networks, I built one by following *Neural Networks From Scratch* by Harrison Kinsley and Daniel Kukiela. Afterward, I began experimenting with various datasets. Although the neural network performed well on simple datasets like MNIST and Iris, it could not classify more complex datasets such as Malaria and CIFAR-10 above a 60% accuracy. Overall, it was incredibly rewarding to see what goes behind neural networks under the hood and understand the workings behind what is sometimes called a "black box." This propelled me to explore more advanced architectures, such as convolutional neural networks.

## Components of the Neural Network
- `Layer_Dense`: Initialize a fully connected layer.
- `Layer_Dropout`: Randomly zeros the outputs of neurons to prevent overfitting by limiting the model's reliance on certain neurons.
- `Activation_ReLU`: Activation applied to the hidden layers (output stays within `[0, ∞)`).
- `Activation_Softmax`: Activation applied to the output (confidence scores) **Multi-class classfication**.
- `Activation_Sigmoid`: Activation applied to the output (`[0,1]` range with 1 being most likely) **Binary classification or Multi-class classification**.
- `Optimizer_SGD`: Stochastic Gradient Descent used to minimize the loss (tune parameters).
- `Optimizer_Adam`: Another widely used Optimizer to minimize the loss (tune parameters).
- `Loss_CatgoricalCrossentropy(Loss)`: Most commonly used loss function paired with `Softmax` to calculate prediction error.
- `Loss_BinaryCrossentropy(Loss)`: Paired with `Sigmoid` and similar to CategoricalCrossentropy but includes the -log of incorrect predictions.


## Requirements
Install any missing dependencies using:
```bash
pip install numpy matplotlib nnfs
```

## Usage
Run the script to run neural network against several datasets!:

```bash
python test_networks.py
```

## Acknowledgments
This project is based on the book *Neural Networks from Scratch* by Harrison Kinsley and Daniel Kukiela.
