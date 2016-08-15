import numpy as np


class Contour:

    # look corners table from
    # https://en.wikipedia.org/wiki/Marching_squares#Isoline
    # corners table is coded as move in clockwise direction
    moves = {
        1: {"to": [1, 0], "from": [0, -1]},   # D
        2: {"to": [0, 1], "from": [1, 0]},   # R
        3: {"to": [0, 1], "from": [0, -1]},   # R
        4: {"to": [-1, 0], "from": [0, 1]},   # U
        6: {"to": [-1, 0], "from": [1, 0]},   # U
        7: {"to": [-1, 0], "from": [0, -1]},   # U
        8: {"to": [0, -1], "from": [-1, 0]},   # L
        9: {"to": [1, 0], "from": [-1, 0]},   # D
        11: {"to": [0, 1], "from": [-1, 0]},  # R
        12: {"to": [0, -1], "from": [0, 1]},  # L
        13: {"to": [1, 0], "from": [0, 1]},  # D
        14: {"to": [0, -1], "from": [1, 0]}   # L
    }

    moves_up = [4, 6, 7]
    moves_down = [1, 9, 13]
    moves_left = [8, 12, 14]
    moves_right = [2, 3, 11]

    from_up = [8, 9, 11]
    from_down = [2, 6, 14]
    from_left = [1, 3, 7]
    from_right = [4, 12, 13]

    def __init__(self, x, y, z):
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.visited_points = None

    def contours(self, thresholds):
        contours = {}
        for t in thresholds:
            points = self.clean(self.find_contours(t))
            if len(points) > 0:
                contours[t] = points
        return contours

    def clean(self, contours):
        """
        Function removes duplicated points
        """
        corrected_contorus = []
        for contour in contours:
            points = np.array(contour)
            to_delete = []
            for i in range(len(points) - 1):
                if np.array_equal(points[i, :], points[i + 1, :]):
                    to_delete.append(i)
            corrected_contorus.append(np.delete(points, to_delete, axis=0))
        return corrected_contorus

    def find_contours(self, threshold):
        contours = []
        bitmap = (self.z > threshold).astype(int)
        self.visited_points = np.zeros(self.z.shape)
        # check if contour start on edge (they have to touches the edge)
        for i in range(bitmap.shape[0] - 1):
            # left
            sq_idx = self.corner_idx(bitmap[i:i+2, 0:2])
            upper = (False if sq_idx != 5 else True)
            if sq_idx in [1, 3, 5, 7] and not self.visited(i, 0, upper):
                contour = self.find_contour_path(bitmap, i, 0, threshold)
                contours.append(contour)
            # right
            sq_idx = self.corner_idx(
                bitmap[i:i+2, bitmap.shape[1]-2:bitmap.shape[1]])
            if sq_idx in [4, 5, 12, 13] and \
                    not self.visited(i, bitmap.shape[1]-2, False):
                contour = self.find_contour_path(
                    bitmap, i, bitmap.shape[1]-2, threshold)
                contours.append(contour)

        for j in range(bitmap.shape[1] - 1):
            # top
            sq_idx = self.corner_idx(bitmap[0:2, j:j+2])
            upper = (False if sq_idx != 10 else True)
            if sq_idx in [8, 9, 10, 11] and not self.visited(0, j, upper):
                contours.append(self.find_contour_path(bitmap, 0, j, threshold))
            # bottom
            sq_idx = self.corner_idx(
                bitmap[bitmap.shape[0]-2:bitmap.shape[0], j:j+2])
            if sq_idx in [2, 6, 10, 14] and \
                    not self.visited(bitmap.shape[0]-2, j, False):
                contour = self.find_contour_path(
                    bitmap, bitmap.shape[0]-2, j, threshold)
                contours.append(contour)

        nonzero_lines = np.nonzero(
            bitmap.shape[1] - np.sum(bitmap[1:-1, :], axis=1))[0] + 1
        # 1:-1 to avoid double check edge

        for i in nonzero_lines:
            for j in range(1, bitmap.shape[1] - 2):
                sq_idx = self.corner_idx(bitmap[i:i+2, j:j+2])
                if sq_idx not in [0, 15] and not self.visited(i, j, False):
                    path = self.find_contour_path(bitmap, i, j, threshold)
                    contours.append(path)
        return contours

    def find_contour_path(self, bitmap, start_i, start_j, threshold):
        i, j = start_i, start_j
        path = [self.to_real_coordinate(
            self.start_point(
                bitmap[i:i+2, j:j+2], np.array([i, j]), threshold))]

        previous_position = None
        step = 0
        while 0 <= i < bitmap.shape[0] - 1 \
                and 0 <= j < bitmap.shape[1] - 1:
            square = bitmap[i:i+2, j:j+2]
            upper = (True if (self.corner_idx(square) in [5, 10] and
                              (previous_position is None or
                               previous_position[0] < i or
                               previous_position[1] < j)) else False)

            if self.visited(i, j, upper):
                # i == start_i and j == start_j and step > 0 and
                break  # cycle

            new_p = self.new_point(
                square, previous_position, np.array([i, j]), threshold)
            path.append(self.to_real_coordinate(new_p))

            self.mark_visited(i, j, upper)
            previous_position_tmp = [i, j]

            i, j = self.new_position(
                square, previous_position, np.array([i, j]))
            previous_position = previous_position_tmp
            step += 1
        return path

    def to_real_coordinate(self, point):
        """
        Parameters
        ----------
        point : list
            List that contains point (x, y) in grid coordinate system

        Returns
        -------
        list
        """
        x_idx = int(point[1])
        y_idx = int(point[0])
        return [self.x[y_idx, x_idx] +
                ((point[1] % 1) * (self.x[y_idx, x_idx + 1] -
                                   self.x[y_idx, x_idx])
                if x_idx + 1 < self.x.shape[1] else 0),
                self.y[y_idx, x_idx] +
                ((point[0] % 1) * (self.y[y_idx + 1, x_idx] -
                                   self.y[y_idx, x_idx])
                if y_idx + 1 < self.x.shape[0] else 0)]

    def new_point(self, sq, previous, position, threshold):
        con_idx = self.corner_idx(sq)
        if con_idx == 5:
            goes_top = ((previous is None and
                         position[1] != self.z.shape[1] - 2) or
                        (previous is not None and
                         (previous[1] + 1 == position[1])))
            heat_from = self.z[position[0] +
                               (0 if goes_top else 1), position[1]]
            heat_to = self.z[position[0] +
                             (0 if goes_top else 1), position[1] + 1]
            pos = position + np.array(
                [(0 if goes_top else 1),
                 self.triangulate(threshold, heat_from, heat_to)])
        elif con_idx == 10:
            goes_right = ((previous is None and
                           position[0] != self.z.shape[0] - 2) or
                          (previous is not None and
                           (previous[0] + 1 == position[0])))
            heat_from = self.z[position[0],
                               position[1] + (1 if goes_right else 0)]
            heat_to = self.z[position[0] + 1,
                             position[1] + (1 if goes_right else 0)]
            pos = position + np.array(
                [self.triangulate(threshold, heat_from, heat_to),
                 (1 if goes_right else 0)])
        else:
            move_dimension = 0 if self.moves[con_idx]['to'][0] == 0 else 1
            pos = (position + np.array(
                self.moves[con_idx]['to']).clip(min=0)).astype(float)
            heat_from = self.z[
                (position[0] + 1 if con_idx in self.moves_down
                 else position[0]),
                (position[1] + 1 if con_idx in self.moves_right
                 else position[1])]
            heat_to = self.z[
                (position[0] if con_idx in self.moves_up else position[0] + 1),
                (position[1] if con_idx in self.moves_left
                 else position[1] + 1)]
            pos[move_dimension] += self.triangulate(
                threshold, heat_from, heat_to)

        return pos.tolist()

    @staticmethod
    def triangulate(threshold, heat_from, heat_to):
        return ((threshold - heat_from) / (heat_to - heat_from)) \
                if heat_from < heat_to else \
                (1 - (threshold - heat_to) / (heat_from - heat_to))

    def start_point(self, sq, position, threshold):
        con_idx = self.corner_idx(sq)
        if con_idx == 5:
            from_left = position[1] != self.z.shape[1] - 2
            heat_from = self.z[position[0],
                               position[1] + (0 if from_left else 1)]
            heat_to = self.z[position[0] + 1,
                             position[1] + (0 if from_left else 1)]
            pos = position + np.array([self.triangulate(
                threshold, heat_from, heat_to),
                0 if from_left else 1])  # left edge 0 every time, same right
        elif con_idx == 10:
            from_top = position[0] != self.z.shape[0] - 2
            heat_from = self.z[position[0] +
                               (0 if from_top else 1), position[1]]
            heat_to = self.z[position[0] +
                             (0 if from_top else 1), position[1] + 1]
            pos = position + np.array(
                [0 if from_top else 1,
                 self.triangulate(threshold, heat_from, heat_to)])
            # on top edge 1 every time, same bottom
        else:
            move_dimension = 0 if self.moves[con_idx]['from'][0] == 0 else 1
            pos = (position + np.array(
                self.moves[con_idx]['from']).clip(min=0)).astype(float)
            heat_from = self.z[
                (position[0] + 1 if con_idx in self.from_down else position[0]),
                (position[1] + 1 if con_idx in self.from_right
                 else position[1])]
            heat_to = self.z[
                (position[0] if con_idx in self.from_up else position[0] + 1),
                (position[1] if con_idx in self.from_left else position[1] + 1)]
            pos[move_dimension] += self.triangulate(
                threshold, heat_from, heat_to)

        return pos.tolist()

    @classmethod
    def new_position(cls, sq, previous, position):
        con_idx = cls.corner_idx(sq)
        if con_idx == 5:
            pos = (position +
                   np.array([(-1 if (previous is None or
                                     previous[1] + 1 == position[1])
                              else 1), 0]))
        elif con_idx == 10:
            pos = (position +
                   np.array([0, (1 if (previous is None or
                                       previous[0] + 1 == position[0])
                                 else -1)]))
        else:
            pos = position + cls.moves[con_idx]['to']
        return pos.tolist()

    @staticmethod
    def corner_idx(sq):
        return np.sum(np.array([[8, 4], [1, 2]]) * sq)

    def visited(self, i, j, upper=True):
        visited = self.visited_points[i, j]
        return (visited in [1, 3] and upper) or (visited >= 2 and not upper)

    def mark_visited(self, i, j, upper=True):
        if not self.visited(i, j, upper):
            self.visited_points[i, j] += (1 if upper else 2)
