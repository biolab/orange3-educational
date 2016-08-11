import unittest

import numpy as np
from numpy.testing import assert_array_equal

from orangecontrib.educational.widgets.utils.contour import Contour


class TestContours(unittest.TestCase):

    def setUp(self):
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        self.xv, self.yv = np.meshgrid(x, y)
        self.z_vertical_asc = self.yv
        self.z_vertical_desc = np.max(self.yv) - self.yv
        self.z_horizontal_asc = self.xv
        self.z_horizontal_desc = np.max(self.xv) - self.xv

        # lt = left top, rt = right top, lb = left bottom, lt = left top
        self.z_rt_lb_desc = self.xv + (np.max(self.yv) - self.yv)
        self.z_rt_lb_asc = (np.max(self.xv) - self.xv) + self.yv
        self.z_lt_rb_asc = self.xv + self.yv
        self.z_lt_rb_desc = (np.max(self.xv) - self.xv) + \
                            (np.max(self.yv) - self.yv)

        # test for testing cycles and 5s and 10s
        self.cycle1 = np.array([[0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0]])
        self.cycle2 = np.array([[0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0]])
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        self.xv_cycle, self.yv_cycle = np.meshgrid(x, y)

    def test_contours(self):
        """
        Test if right amount of values
        """
        c = Contour(self.xv, self.yv, self.z_vertical_asc)
        c_lines = c.contours([1, 2, 3])

        # all line exists in particular data
        self.assertIn(1, c_lines.keys())
        self.assertIn(2, c_lines.keys())
        self.assertIn(3, c_lines.keys())

        # in particular data none line are in more peaces
        self.assertEqual(len(c_lines[1]), 1)
        self.assertEqual(len(c_lines[2]), 1)
        self.assertEqual(len(c_lines[3]), 1)

        c = Contour(self.xv, self.yv, self.z_vertical_desc)
        c_lines = c.contours([1, 2, 3])

        # all line exists in particular data
        self.assertIn(1, c_lines.keys())
        self.assertIn(2, c_lines.keys())
        self.assertIn(3, c_lines.keys())

        # in particular data none line are in more peaces
        self.assertEqual(len(c_lines[1]), 1)
        self.assertEqual(len(c_lines[2]), 1)
        self.assertEqual(len(c_lines[3]), 1)

        c = Contour(self.xv, self.yv, self.z_horizontal_asc)
        c_lines = c.contours([1, 2, 3])

        # all line exists in particular data
        self.assertIn(1, c_lines.keys())
        self.assertIn(2, c_lines.keys())
        self.assertIn(3, c_lines.keys())

        # in particular data none line are in more peaces
        self.assertEqual(len(c_lines[1]), 1)
        self.assertEqual(len(c_lines[2]), 1)
        self.assertEqual(len(c_lines[3]), 1)

        c = Contour(self.xv, self.yv, self.z_horizontal_desc)
        c_lines = c.contours([1, 2, 3])

        # all line exists in particular data
        self.assertIn(1, c_lines.keys())
        self.assertIn(2, c_lines.keys())
        self.assertIn(3, c_lines.keys())

        # in particular data none line are in more peaces
        self.assertEqual(len(c_lines[1]), 1)
        self.assertEqual(len(c_lines[2]), 1)
        self.assertEqual(len(c_lines[3]), 1)

        c = Contour(self.xv, self.yv, self.z_lt_rb_asc)
        c_lines = c.contours([1, 2, 3])

        # all line exists in particular data
        self.assertIn(1, c_lines.keys())
        self.assertIn(2, c_lines.keys())
        self.assertIn(3, c_lines.keys())

        # in particular data none line are in more peaces
        self.assertEqual(len(c_lines[1]), 1)
        self.assertEqual(len(c_lines[2]), 1)
        self.assertEqual(len(c_lines[3]), 1)

        c = Contour(self.xv, self.yv, self.z_lt_rb_desc)
        c_lines = c.contours([1, 2, 3])

        # all line exists in particular data
        self.assertIn(1, c_lines.keys())
        self.assertIn(2, c_lines.keys())
        self.assertIn(3, c_lines.keys())

        # in particular data none line are in more peaces
        self.assertEqual(len(c_lines[1]), 1)
        self.assertEqual(len(c_lines[2]), 1)
        self.assertEqual(len(c_lines[3]), 1)

        c = Contour(self.xv, self.yv, self.z_rt_lb_asc)
        c_lines = c.contours([1, 2, 3])

        # all line exists in particular data
        self.assertIn(1, c_lines.keys())
        self.assertIn(2, c_lines.keys())
        self.assertIn(3, c_lines.keys())

        # in particular data none line are in more peaces
        self.assertEqual(len(c_lines[1]), 1)
        self.assertEqual(len(c_lines[2]), 1)
        self.assertEqual(len(c_lines[3]), 1)

        c = Contour(self.xv, self.yv, self.z_rt_lb_desc)
        c_lines = c.contours([1, 2, 3])

        # all line exists in particular data
        self.assertIn(1, c_lines.keys())
        self.assertIn(2, c_lines.keys())
        self.assertIn(3, c_lines.keys())

        # in particular data none line are in more peaces
        self.assertEqual(len(c_lines[1]), 1)
        self.assertEqual(len(c_lines[2]), 1)
        self.assertEqual(len(c_lines[3]), 1)

        # test in cycle set
        c = Contour(self.xv_cycle, self.yv_cycle, self.cycle1)
        c_lines = c.contours([0.5])

        self.assertIn(0.5, c_lines.keys())
        self.assertEqual(len(c_lines[0.5]), 1)

        # test start with square 5, before only 10 was checked
        c = Contour(self.xv_cycle, self.yv_cycle, self.cycle2)
        c_lines = c.contours([0.5])

        self.assertIn(0.5, c_lines.keys())
        self.assertEqual(len(c_lines[0.5]), 1)

        # test no contours, then no key in dict
        c = Contour(self.xv_cycle, self.yv_cycle, self.cycle2)
        c_lines = c.contours([1.5])

        self.assertNotIn(1.5, c_lines.keys())

    def test_find_contours(self):
        """
        Test if right contours found for threshold
        """
        # check all horizontal edges
        c = Contour(self.xv, self.yv, self.z_horizontal_asc)

        points = c.find_contours(1)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([1, i], points[0])

        points = c.find_contours(5)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([5, i], points[0])

        c = Contour(self.xv, self.yv, self.z_horizontal_desc)

        points = c.find_contours(1)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([9, i], points[0])

        points = c.find_contours(5)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([5, i], points[0])

        # check all vertical edges
        c = Contour(self.xv, self.yv, self.z_vertical_asc)

        points = c.find_contours(1)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([i, 1], points[0])

        points = c.find_contours(5)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([i, 5], points[0])

        c = Contour(self.xv, self.yv, self.z_vertical_desc)

        points = c.find_contours(1)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([i, 9], points[0])

        points = c.find_contours(5)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([i, 5], points[0])

        # check all top-left bottom-right edges
        c = Contour(self.xv, self.yv, self.z_lt_rb_asc)

        points = c.find_contours(1)
        self.assertEqual(len(points), 1)  # only one line in particular example
        self.assertIn([0, 1], points[0])
        self.assertIn([1, 0], points[0])

        points = c.find_contours(10)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([i, 10-i], points[0])

        c = Contour(self.xv, self.yv, self.z_lt_rb_desc)

        points = c.find_contours(1)
        self.assertEqual(len(points), 1)  # only one line in particular example
        self.assertIn([10, 9], points[0])
        self.assertIn([9, 10], points[0])

        points = c.find_contours(10)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([i, 10-i], points[0])

        # check all top-right bottom-left edges
        c = Contour(self.xv, self.yv, self.z_rt_lb_asc)

        points = c.find_contours(1)
        self.assertEqual(len(points), 1)  # only one line in particular example
        self.assertIn([9, 0], points[0])
        self.assertIn([10, 1], points[0])

        points = c.find_contours(10)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([10-i, 10-i], points[0])

        c = Contour(self.xv, self.yv, self.z_rt_lb_desc)

        points = c.find_contours(1)
        self.assertEqual(len(points), 1)  # only one line in particular example
        self.assertIn([0, 9], points[0])
        self.assertIn([1, 10], points[0])

        points = c.find_contours(10)
        self.assertEqual(len(points), 1)  # only one line in particular example
        for i in range(11):
            self.assertIn([10-i, 10-i], points[0])

        c = Contour(self.xv_cycle, self.yv_cycle, self.cycle1)

        points = c.find_contours(0.5)
        self.assertEqual(len(points[0]), 13)
        self.assertIn([1, 0.5], points[0])
        self.assertIn([1.5, 1], points[0])
        self.assertIn([2, 1.5], points[0])
        self.assertIn([2.5, 2], points[0])
        self.assertIn([2, 2.5], points[0])
        self.assertIn([1.5, 3], points[0])
        self.assertIn([1, 3.5], points[0])
        self.assertIn([0.5, 3], points[0])
        self.assertIn([1, 2.5], points[0])
        self.assertIn([1.5, 2], points[0])
        self.assertIn([1, 1.5], points[0])
        self.assertIn([0.5, 1], points[0])

        c = Contour(self.xv_cycle, self.yv_cycle, self.cycle2)

        points = c.find_contours(0.5)
        self.assertEqual(len(points[0]), 13)
        self.assertIn([2, 0.5], points[0])
        self.assertIn([2.5, 1], points[0])
        self.assertIn([2, 1.5], points[0])
        self.assertIn([1.5, 2], points[0])
        self.assertIn([2, 2.5], points[0])
        self.assertIn([2.5, 3], points[0])
        self.assertIn([2, 3.5], points[0])
        self.assertIn([1.5, 3], points[0])
        self.assertIn([1, 2.5], points[0])
        self.assertIn([0.5, 2], points[0])
        self.assertIn([1, 1.5], points[0])
        self.assertIn([1.5, 1], points[0])

    def test_to_real_coordinate(self):
        c = Contour(self.xv, self.yv, self.z_horizontal_asc)

        # integers same because of grid with integers
        self.assertEqual(c.to_real_coordinate([1, 1]), [1, 1])

        # coordinate have to have x on first place (before row first)
        self.assertEqual(c.to_real_coordinate([1, 2]), [2, 1])

        # middle values
        self.assertEqual(c.to_real_coordinate([1, 1.5]), [1.5, 1])
        self.assertEqual(c.to_real_coordinate([1.5, 1.5]), [1.5, 1.5])
        self.assertEqual(c.to_real_coordinate([1.5, 1]), [1, 1.5])
        self.assertEqual(c.to_real_coordinate([5, 5.5]), [5.5, 5])
        self.assertEqual(c.to_real_coordinate([5.5, 5.5]), [5.5, 5.5])
        self.assertEqual(c.to_real_coordinate([5.5, 5]), [5, 5.5])

        # meshgrid no integers
        xv, yv = np.meshgrid(np.linspace(0, 5, 11), np.linspace(0, 5, 11))
        c = Contour(xv, yv, self.z_horizontal_asc)

        self.assertEqual(c.to_real_coordinate([1, 1]), [0.5, 0.5])
        self.assertEqual(c.to_real_coordinate([1, 1.5]), [0.75, 0.5])
        self.assertEqual(c.to_real_coordinate([1.5, 1.5]), [0.75, 0.75])
        self.assertEqual(c.to_real_coordinate([1.5, 1]), [0.5, 0.75])
        self.assertEqual(c.to_real_coordinate([5, 5.5]), [2.75, 2.5])
        self.assertEqual(c.to_real_coordinate([5.5, 5.5]), [2.75, 2.75])
        self.assertEqual(c.to_real_coordinate([5.5, 5]), [2.5, 2.75])

    def test_triangulate(self):
        self.assertEqual(Contour.triangulate(0, 0, 1), 0)
        self.assertEqual(Contour.triangulate(1, 0, 1), 1)
        self.assertEqual(Contour.triangulate(0.5, 0, 1), 0.5)
        self.assertEqual(Contour.triangulate(0.3, 0, 1), 0.3)

        self.assertEqual(Contour.triangulate(0, 1, 0), 1)
        self.assertEqual(Contour.triangulate(1, 1, 0), 0)
        self.assertEqual(Contour.triangulate(0.5, 1, 0), 0.5)
        self.assertEqual(Contour.triangulate(0.3, 1, 0), 0.7)

    def test_new_position(self):
        # when sq not equal 5 or 10 previous position does not matter
        assert_array_equal(Contour.new_position(
            np.array([[0, 0], [1, 0]]), None, np.array([1, 1])), [2, 1])
        assert_array_equal(Contour.new_position(
            np.array([[0, 0], [0, 1]]), None, np.array([1, 1])), [1, 2])
        assert_array_equal(Contour.new_position(
            np.array([[0, 0], [1, 1]]), None, np.array([1, 1])), [1, 2])
        assert_array_equal(Contour.new_position(
            np.array([[0, 1], [0, 0]]), None, np.array([1, 1])), [0, 1])
        assert_array_equal(Contour.new_position(
            np.array([[0, 1], [0, 1]]), None, np.array([1, 1])), [0, 1])
        assert_array_equal(Contour.new_position(
            np.array([[0, 1], [1, 1]]), None, np.array([1, 1])), [0, 1])
        assert_array_equal(Contour.new_position(
            np.array([[1, 0], [0, 0]]), None, np.array([1, 1])), [1, 0])
        assert_array_equal(Contour.new_position(
            np.array([[1, 0], [1, 0]]), None, np.array([1, 1])), [2, 1])
        assert_array_equal(Contour.new_position(
            np.array([[1, 0], [1, 1]]), None, np.array([1, 1])), [1, 2])
        assert_array_equal(Contour.new_position(
            np.array([[1, 1], [0, 0]]), None, np.array([1, 1])), [1, 0])
        assert_array_equal(Contour.new_position(
            np.array([[1, 1], [1, 0]]), None, np.array([1, 1])), [2, 1])
        assert_array_equal(Contour.new_position(
            np.array([[1, 1], [0, 1]]), None, np.array([1, 1])), [1, 0])

        # sq = 5
        # start on edge
        assert_array_equal(Contour.new_position(
            np.array([[0, 1], [1, 0]]), None, np.array([1, 1])), [0, 1])
        # previous from left
        assert_array_equal(Contour.new_position(
            np.array([[0, 1], [1, 0]]), np.array([1, 0]),
            np.array([1, 1])), [0, 1])
        # previous from right
        assert_array_equal(Contour.new_position(
            np.array([[0, 1], [1, 0]]), np.array([1, 2]),
            np.array([1, 1])), [2, 1])

        # sq = 10
        # start on edge
        assert_array_equal(Contour.new_position(
            np.array([[1, 0], [0, 1]]), None, np.array([1, 1])), [1, 2])
        # previous from top
        assert_array_equal(Contour.new_position(
            np.array([[1, 0], [0, 1]]), np.array([0, 1]),
            np.array([1, 1])), [1, 2])
        # previous from bottom
        assert_array_equal(Contour.new_position(
            np.array([[1, 0], [0, 1]]), np.array([2, 1]),
            np.array([1, 1])), [1, 0])

    def test_corner_idx(self):
        self.assertEqual(Contour.corner_idx([[0, 0], [0, 0]]), 0)
        self.assertEqual(Contour.corner_idx([[0, 0], [1, 0]]), 1)
        self.assertEqual(Contour.corner_idx([[0, 0], [0, 1]]), 2)
        self.assertEqual(Contour.corner_idx([[0, 0], [1, 1]]), 3)
        self.assertEqual(Contour.corner_idx([[0, 1], [0, 0]]), 4)
        self.assertEqual(Contour.corner_idx([[0, 1], [1, 0]]), 5)
        self.assertEqual(Contour.corner_idx([[0, 1], [0, 1]]), 6)
        self.assertEqual(Contour.corner_idx([[0, 1], [1, 1]]), 7)
        self.assertEqual(Contour.corner_idx([[1, 0], [0, 0]]), 8)
        self.assertEqual(Contour.corner_idx([[1, 0], [1, 0]]), 9)
        self.assertEqual(Contour.corner_idx([[1, 0], [0, 1]]), 10)
        self.assertEqual(Contour.corner_idx([[1, 0], [1, 1]]), 11)
        self.assertEqual(Contour.corner_idx([[1, 1], [0, 0]]), 12)
        self.assertEqual(Contour.corner_idx([[1, 1], [1, 0]]), 13)
        self.assertEqual(Contour.corner_idx([[1, 1], [0, 1]]), 14)
        self.assertEqual(Contour.corner_idx([[1, 1], [1, 1]]), 15)

    def test_visited(self):
        c = Contour(self.xv, self.yv, self.z_rt_lb_desc)
        c.visited_points = np.zeros(self.xv.shape)

        self.assertFalse(c.visited(0, 0, True))
        self.assertFalse(c.visited(0, 0, False))

        # check if upper
        c.mark_visited(0, 0, True)
        self.assertTrue(c.visited(0, 0, True))
        self.assertFalse(c.visited(0, 0, False))

        # check if lower
        c.mark_visited(1, 1, False)
        self.assertFalse(c.visited(1, 1, True))
        self.assertTrue(c.visited(1, 1, False))

        # check if ok when mark again
        c.mark_visited(1, 1, False)
        self.assertFalse(c.visited(1, 1, True))
        self.assertTrue(c.visited(1, 1, False))

        c.mark_visited(0, 0, True)
        self.assertTrue(c.visited(0, 0, True))
        self.assertFalse(c.visited(0, 0, False))

        # check if booth lower fist, and upper first
        c.mark_visited(1, 1, True)
        self.assertTrue(c.visited(1, 1, True))
        self.assertTrue(c.visited(1, 1, False))

        c.mark_visited(0, 0, False)
        self.assertTrue(c.visited(0, 0, True))
        self.assertTrue(c.visited(0, 0, False))
