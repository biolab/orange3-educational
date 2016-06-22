Educational k-means
===================

.. figure:: icons/mywidget.png

Educational widgets that shows working of a k-means clustering.

Signals
-------
Inputs
~~~~~~
- **Data**
Input data set.

Outputs
~~~~~~~

- **Data**
Data set with clusters labels annotation.
- **Centroids**
Centroids position

Description
-----------

The aim of this widget is to show the working of a k-means clustering algorithm on two attributes from data set.
Widget applies k-means clustering to the selected two attributes step by step. User can step through the algorithm and
see how the algorithm works.

1. Select attribute for **x** axis and attribute for **y** axis.

2. Select number of centroids in the spinner. If you want new random positions of the centroids or restart the algorithm
in any point, you can click on **Randomize** button. If you want to add centroid on particular position in the graph
just click on this position. If you want to move centroid, grab it and drop it on the desired position.

3. Step through the algorithm with **Recompute centroids** and **Reassign membership**. If you want to make step back
use **Step back** button. You can also step automatically with pressing on **Run** button. **Speed** spinner can be used
to set the speed of automatic stepping.




