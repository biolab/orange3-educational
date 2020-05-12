import unittest
from Orange.data import Table, Domain, ContinuousVariable
from orangecontrib.educational.widgets.utils.polynomialtransform \
    import PolynomialTransform


class TestPolynomialTransform(unittest.TestCase):

    def setUp(self):
        self.titanic = Table('titanic')
        self.iris = Table('iris')
        domain = Domain([ContinuousVariable("a"), ContinuousVariable("b")])
        self.data = Table.from_list(
            domain, [[1.3, 2], [3, 4.1], [1, 2], [3.2, 3.1]]
        )
        # data with two columns

    def test_plynomial_transform(self):
        """
        Test polynomial transformation widget
        """

        # check if limited for not sufficient data
        polynomial_transform = PolynomialTransform(1)

        # titanic does not have all continuous attributes
        self.assertRaises(ValueError, polynomial_transform, self.titanic)

        # check when too much continous attributes
        self.assertRaises(ValueError, polynomial_transform, self.iris)

        # check if number of cell columns sufficient
        no_columns = 2  # sum of pascal triangle
        for i in range(1, 10):
            polynomial_transform = PolynomialTransform(i)
            self.assertEqual(
                len(polynomial_transform(self.data).domain.attributes),
                no_columns)
            no_columns += (i + 2)

        # check if transformation sufficient
        polynomial_transform = PolynomialTransform(5)
        transformed_data = polynomial_transform(self.data)

        for i, row in enumerate(transformed_data):
            self.assertEqual(row[0], self.data[i][0])
            self.assertEqual(row[1], self.data[i][1])

            self.assertEqual(row[2], self.data[i][0] ** 2)
            self.assertEqual(row[3], self.data[i][0] * self.data[i][1])
            self.assertEqual(row[4], self.data[i][1] ** 2)

            self.assertEqual(row[5], self.data[i][0] ** 3)
            self.assertEqual(row[6], self.data[i][0] ** 2 * self.data[i][1])
            self.assertEqual(row[7], self.data[i][0] * self.data[i][1] ** 2)
            self.assertEqual(row[8], self.data[i][1] ** 3)

            self.assertEqual(row[9], self.data[i][0] ** 4)
            self.assertEqual(row[10], self.data[i][0] ** 3 * self.data[i][1])
            self.assertEqual(
                row[11], self.data[i][0] ** 2 * self.data[i][1] ** 2)
            self.assertEqual(row[12], self.data[i][0] * self.data[i][1] ** 3)
            self.assertEqual(row[13], self.data[i][1] ** 4)

            self.assertEqual(row[14], self.data[i][0] ** 5)
            self.assertEqual(row[15], self.data[i][0] ** 4 * self.data[i][1])
            self.assertEqual(
                row[16], self.data[i][0] ** 3 * self.data[i][1] ** 2)
            self.assertEqual(
                row[17], self.data[i][0] ** 2 * self.data[i][1] ** 3)
            self.assertEqual(row[18], self.data[i][0] * self.data[i][1] ** 4)
            self.assertEqual(row[19], self.data[i][1] ** 5)
