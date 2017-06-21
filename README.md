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

**Logistic Classifier:**

![equation](http://www.sciweavers.org/tex2img.php?eq=W%20X%20%2B%20b%20%3D%20y&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

* W - weights
* X - input vector
* b - BIAS (threshold value)

**Softmax:**

![equation](http://www.sciweavers.org/tex2img.php?eq=S%20%5Cbig%28y_i%5Cbig%29%20%3D%20%20%5Cfrac%7Be%5E%7By_i%7D%7D%7B%20%5Csum_j%20e%5E%7By_j%7D%20%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

Function for calculating probabilites of scores (e.g. input vector).
Example:

![equation](http://www.sciweavers.org/tex2img.php?eq=y%20%5Cbegin%7Bbmatrix%7D2.0%20%5Crightarrow%20p%20%3D%200.7%5C%5C1.0%20%5Crightarrow%20p%20%3D%200.2%20%5C%5C0.1%20%5Crightarrow%20p%20%3D%200.1&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

scores = logits
