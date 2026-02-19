# Neural Network

Neural network is loosely inspired by human brain, that uses interconnected layers of algorithms (nodes) to process datam recognize complex patterns and make predictions.



```css
Input Layer → Hidden Layer(s) → Output Layer
   data          thinking         decision
```

## 1. Input Layer

🔹 What is the Input Layer?

The Input Layer is the first layer of a neural network. Its ___job is not to think or decide___ — its job is simply to receive data and pass it forward.

Whatever data you want the model to learn from **must enter through the input layer.**


🔹 What does it actually contain?

Each neuron in the input layer represents one feature of the data.

Examples:

    🔹Image recognition → each neuron = one pixel value
    🔹House price prediction → neurons = size, location, number of rooms
    🔹Text processing → neurons = encoded words or characters

## 2. Forward Pass (_Prediction_)

```css
Input → Hidden Layers → Output
```
The network uses its current knowledge (weights & biases) to produce a prediction.

1. Input data enters the network
2. Each layer processes it

> “Forward pass = guess.”

## 3. Backward Propagation (_Correction_)

```css
Forward pass → Error → Backprop → Update → Repeat
```
Backpropagation is how the network learns from its mistake.

It:

1. **Compares** the prediction to the correct answer
2. Measures the **error**
3. Sends that error backward
4. Adjusts weights and biases to do better next time

```txt 
Direction matters
🔹 Forward pass → left to right
🔹 Backpropagation → right to left
```

What changes?

Only **Weights and biases** are changed

**Input** data _does not_ change

**Network Structure** _does not_ change

> “Backpropagation = fix the weights.”

Example:
``` css
        Input →   →   →   Newron (Forward Pass)
                                    ↓
    (Adjust Weight and bias)  →  TRAIN (Multiple layer of neurons)
          ↑                         ↓
        ERROR ←     ←   ←   ←   Predicts  
```

Data is fed to the neurons -> Prediction $3,000 -> Actual Price $3,500 -> Error Found _Too low by $500_ -> Backpropagation _Adjust weight to Increase future prediction_.  

## 4. Linear regression

It finds the best straufgt kine to describe the relationship between teo things such as if house is big price is high, or if house is small price is low. 

It learns 
- How strong is the relation
- Direction of the relation

**Principal**

1. Predict a line
2. evaluate margin of error 
3. Adjust the line
4. Repeat until it fits well

> “Linear regression = best straight-line guess.”

- Linear regression finds a straight-line relationship
- It predicts numbers, not categories
- It learns by reducing prediction error
- It’s the foundation of machine learning


## 5. ACTIVATION Functions

Activation functions are essential mathematical equations in neural networks that introduce non-linearity to the model, enabling it to learn complex, non-linear patterns in data. Without them, deep neural networks would act as simple linear models. They determine if a neuron should activate ("fire") by mapping inputs to specific outputs, crucial for tasks like image recognition, NLP, and decision-making.

$WX+B $

Where `W = Weight`, `X = Input`, `B = Bias`

Activation functions are used hidden layers.

```css
Weighted sum → Activation → Output
   numbers       decision    signal
```
> “Activation = decision maker inside a neuron.”

> Activation functions add non-linearity and decide how strongly a neuron fires.

Basic operation:
- Neuron receives numbers
- Adds them up (with weights)
- Activation function asks:
    
    “Is this signal important enough?”

Then it outputs:
- 0 (ignore)
- some value (pass it on)


🔹ReLU - **_Rectified Linear Unit_** ($f(x) = max(0,x)$) 

It is the most popular default, it speeds up convergence and mitigates the vanishing gradient problem. _Used in hidden layers_
    
**Principal**: Takes only positive value, discard all 0 values

🔹Leaky ReLU - 
    
Allows a small, non-zero gradient when the input is negative, fixing the `"dying ReLU"` problem.
    
**Principal**: Not ignoring negative values instead it will multiply with near zero value 0.01 (ALPHA)

🔹Sigmoid / Logistic ($f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$) or ($f(x) = \frac{1}{1 + e^{-x}}$)
    
Squashes values between 0 and 1. Used mainly for binary classification.

🔹Tanh ___Hyperbolic Tangent___ 
    
Outputs values between -1 and 1, often preferred over `sigmoid` for hidden layers.

🔹Softmax : 
    
Used in the output layer for multi-class classification to turn logits into probabilities that sum to 1.


> Note: Leaky ReLU is Importent in Deep learning models. _There are several simimlar model `Tanh`, `sigmoid`, `softmx`_. 

> __Except `Sigmoid` / `Softmx` other activation functions are used in input or hidden layers.__

## 6. Output layer Decesion making functions:
**Sigmoid** - Binary Classification any output data more than 0.5 and near 1 is considered for output

**Softmx** - For Binary Categorical Data. as it will replace anything more then 0.5 with 1

Optimization of a model is done based on gradient descent

Backpropagation er maddhome loss k komanor jonno weight optimize kora hoi gradient descent (Stochastic gradient descent, ADAM - Adaptive moment estimation) 

ADAM Optimizer

### Review

> “Forward pass = guess.”

> “Backpropagation = fix the weights.”

> “Linear regression = best straight-line guess.”

> “Activation = decision maker inside a neuron.”

> Activation functions add non-linearity and decide how strongly a neuron fires.

> **ReLU** (Default) takes only positive value, discard all 0 values

> **Leaky ReLU** is Importent in Deep learning models

> Except `Sigmoid` / `Softmx` other activation functions are used in input or hidden layers.