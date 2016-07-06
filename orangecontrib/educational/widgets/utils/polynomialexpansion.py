from Orange.preprocess.preprocess import Preprocess
from Orange.data.domain import DiscreteVariable, ContinuousVariable
from Orange.data import Table, Domain
from sklearn.preprocessing import PolynomialFeatures

class PolynomialTransform(Preprocess):

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, data):
        """
        Generate a new feature matrix consisting of all polynomial combinations of the features.


        Parameters
        ----------
        data

        Returns
        -------

        """

        if sum(isinstance(x, DiscreteVariable) for x in data.domain.attributes) > 0:
            raise Exception('Not all attributes are Continuous. PolynomialExpansion works only on continuous features.')

        poly = PolynomialFeatures(self.degree)
        poly_x = poly.fit_transform(data.X)[:, 1:] # this function appends ones to the beginning
        poly_domain = Domain(
            data.domain.attributes + tuple(ContinuousVariable("Feature%d" % i)
                                           for i in range(len(data.domain.attributes), poly_x.shape[1])),
            data.domain.class_var,
            data.domain.metas
        )
        print("kaaaaa")
        return Table(poly_domain, poly_x, data.Y, data.metas)