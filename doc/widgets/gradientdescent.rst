Gradient Descent
================

.. figure:: icons/gradient-descent.png

Educational widget that shows gradient descent algorithm on a logistic regression.

Signals
-------

**Inputs**:

- **Data**

Input data set.

**Outputs**:

- **Data**

Data with columns selected in widget.

- **Classifier**

Model produced on the current step of the algorithm.

- **Coefficients**

Logistic regression coefficients on the current step of the algorithm.

Description
-----------

This widget shows steps of `gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`__ for logistic regression
step by step. Gradient descent is demonstrated on two attributes that are selected by user.

.. figure:: images/gradient-descent-stamped.png

1. Select two attributes (**x** and **y**) on which gradient descent algorithm is preformed.
   Select **target class**. It is class that is classified against all other classes.

2. **Learning rate** is step size in a gradient descent

   With **stochastic** checkbox you can select whether gradient descent is
   `stochastic <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`__ or not.
   If stochastic is checked you can set **step size** that is amount of steps of stochastic gradient descent
   performed in one press on step button.

   **Restart**: start algorithm from beginning

3. **Step**: perform one step of the algorithm

   **Step back**: make a step back in the algorithm

4. **Run**: automatically perform several steps until algorithm converge

   **Speed**: set speed of automatic stepping

5. **Save Image** saves the image to the computer in a .svg or .png
   format.

   **Report** includes widget parameters and visualization in the report.

Example
-------

In Orange we connected *File* widget with *Iris* data set to *Gradient Descent* widget. We connected outputs of
the widget to *Predictions* widget to see how data are classified and *Data Table* widget where we inspect coefficients
of logistic regression.

.. figure:: images/gradient-descent-flow.png

We opened *Gradient Descent* widget and set *X* to *sepal width* and *Y* to *sepal length*. Target class is set to
*Iris-virginica*. We set *learning rate* to 0.02. With click in graph we set beginning coefficients (red dot).

.. figure:: images/gradient-descent1.png

We performs step of the algorithm with pressing **Step** button. When we get bored with clicking we can finish stepping
with press on **Run** button.

.. figure:: images/gradient-descent2.png

If we want to go back in the algorithm we can do it with pressing **Step back** button. This will also change model.
Current model uses positions of last coefficients (red-yellow dot).

.. figure:: images/gradient-descent3.png

In the end we want to see predictions for input data so we can open *Predictions* widget. Predictions are listed in
left column. We can compare this predictions to real classes.

.. figure:: images/gradient-descent4.png
