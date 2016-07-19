import numpy as np
from Orange.data.domain import DiscreteVariable, ContinuousVariable
from Orange.data import Table, Domain, Instance
from Orange.preprocess.transformation import Identity
from Orange.preprocess.preprocess import Preprocess

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
            raise ValueError('Not all attributes are Continuous. '
                             'PolynomialExpansion works only on continuous features.')
        if len(data.domain.attributes) > 2:
            raise ValueError('Too much attributes')

        vars = data.domain.attributes
        poly_vars = list(vars)

        for i in range(2, self.degree + 1):
            for j in range(i + 1):
                p1, p2 = i - j, j
                poly_vars.append(
                    ContinuousVariable(
                        "{n1} ^ {p1} * {n2} ^ {p2}".format(n1=vars[0].name,
                                                           n2=vars[1].name,
                                                           p1=p1,
                                                           p2=p2),
                        compute_value=MultiplyAndPower(vars, p1, p2)))

        domain = Domain(poly_vars, data.domain.class_var, data.domain.metas)
        return data.from_table(domain, data)


class Transformation2:
    """
    Base class for simple transformations of multiple variables into one.
    """
    def __init__(self, variables):
        """
        :param variable: The variable whose transformed value is returned.
        :type variable: int or str or :obj:`~Orange.data.Vardiable`
        """
        self.variables = variables

    def __call__(self, data):
        """
        Return transformed column from the data by extracting the column view
        from the data and passing it to the `transform` method.
        """
        data_all = []
        inst = isinstance(data, Instance)
        for i, var in enumerate(self.variables):
            try:
                attr_index = data.domain.index(var)
            except ValueError:
                if var.compute_value is None:
                    raise ValueError("{} is not in domain".
                                     format(self.variables[i].name))
                attr_index = None
            if attr_index is None:
                data_col = var.compute_value(data)
            elif inst:
                data_col = np.array([float(data[attr_index])])
            else:
                data_col = data.get_column_view(attr_index)[0]
            data_all.append(data_col)
        transformed_col = self.transform(data_all)
        if inst and isinstance(transformed_col, np.ndarray) and transformed_col.shape:
            transformed_col = transformed_col[0]
        return transformed_col

    def transform(self, c):
        """
        Return the transformed value of the argument `c`, which can be a number
        of a vector view.
        """
        raise NotImplementedError(
            "ColumnTransformations must implement method 'transform'.")


class MultiplyAndPower(Transformation2):
    """
    Return an indicator value that equals 1 if the variable has the specified
    value and 0 otherwise.
    """
    def __init__(self, variables, power1, power2):
        """
        :param variable: The variable whose transformed value is returned.
        :type variable: int or str or :obj:`~Orange.data.Variable`

        :param value: The value to which the indicator refers
        :type value: int or float
        """
        super().__init__(variables)
        self.power1 = power1
        self.power2 = power2

    def transform(self, c):
        return (c[0] ** self.power1) * (c[1] ** self.power2)