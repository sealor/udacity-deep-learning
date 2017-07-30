# Links

* https://classroom.udacity.com
* [MathML Tutorial](http://www.math-it.org/Publikationen/MathML_de.html)
* [MathML Online Editor](https://www.tutorialspoint.com/mathml/try_mathml.php)

Python:

* [numpy Array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)
* [numpy Exp](https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html)

# Installation

* [IPython](https://wiki.ubuntuusers.de/Archiv/IPython/)
* `apt-get install ipython3`
* toggle auto indentation: `%autoindent`

# 9. Training your logistic classifier

**Logistic Classifier**

![equation](http://mathurl.com/y9e58teg.png)

* W - weights
* X - input vector
* b - BIAS (threshold value)

**Softmax**

![equation](http://mathurl.com/ybtfplbf.png)

Function for calculating probabilities of scores (e.g. input vector).
Example:

![score example 1](http://mathurl.com/ybgqp6ud.png)

scores = logits

When all scores are multiplied with 10, the probabilities get close to either 0.0 or 1.0.
When all scores are divided by 10, the probabilities get close to the uniform distribution.

![score example 2](http://mathurl.com/y9z6pb7j.png)

**One-Hot Encoding**

For encoding labels only one 1.0 value is used in the input vector. All other values are 0.0.

![one-hot encoding example](http://mathurl.com/yb2bs7me.png)

**Cross-Entropy**

Cross-Entropy measures the distance between two vectors. For example the result vector of the classification and the vactor that corresponds with the labels.

![cross entropy](http://mathurl.com/ycerht8k.png)

**Multinomial Logistic Classification**

1. all inputs are calculated to score values with the **Logistic Classifier**
2. the resulting scores are new used for calculating probabilities using the **Softmax** function
3. those probabilities can then be used to calculate the distance to the label vectors with the **Cross-Entropy**

![Multinomial logistic classification](http://mathurl.com/y9av44od.png)

**Minimising the Cross Entropy**

For getting a good Logistic Classifier we need to minimise the **Average Cross-Entropy**.

![Loss function](http://mathurl.com/y9rn2ekw.png)

Loss = Average Cross-Entropy

**Normalized inputs**

Because of numerical instability of floating point operations we normalize all inputs.

**Assignment 1: notMNIST**

[IPython Assignment 1: notMNIST](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb)

**Validation Test Set Size**

* `>` 30.000 examples
* changes in `>` 0.1% in accuracy

**Stochastic Gradient Descent**

* random data for error calculation (0.1%)
* very small steps for Gradient Descent

**Momentum and Learning Rate Decay**

* Momentum: running average over Gradient Decent
* Learning Rate Decay:
  * lower step size over time
  * lower step size when reaching a plateau

**Parameter Hyperspace**

* use small leanring rate
* Stochastic Gradient Descent "black magic"
  * initial learning rate
  * learning rate decay
  * momentum
  * batch size
  * weight initialization
* AdaGrad is adaptive algorithm for learning rate and momentum
