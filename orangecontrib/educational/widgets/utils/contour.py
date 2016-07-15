import numpy as np


class Contour:

    # look corners table from https://en.wikipedia.org/wiki/Marching_squares#Isoline
    # corners table is coded as move in clockwise direction
    corners = {
        1: {"to": [1, 0.5], "from": [0.5, 0]},   # D
        2: {"to": [0.5, 1], "from": [1, 0.5]},   # R
        3: {"to": [0.5, 1], "from": [0.5, 0]},   # R
        4: {"to": [0, 0.5], "from": [0.5, 1]},   # U
        6: {"to": [0, 0.5], "from": [1, 0.5]},   # U
        7: {"to": [0, 0.5], "from": [0.5, 0]},   # U
        8: {"to": [0.5, 0], "from": [0, 0.5]},   # L
        9: {"to": [1, 0.5], "from": [0, 0.5]},   # D
        11: {"to": [0.5, 1], "from": [0, 0.5]},  # R
        12: {"to": [0.5, 0], "from": [0.5, 1]},  # L
        13: {"to": [1, 0.5], "from": [0.5, 1]},  # D
        14: {"to": [0.5, 0], "from": [1, 0.5]}   # L
    }

    def __init__(self, x, y, z):
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.visited = None

    def contours(self, thresholds):
        contours = {}
        for t in thresholds:
            points = self.find_contours(t)
            if len(points) > 0:
                contours[t] = points
        return contours

    def find_contours(self, threshold):
        print(threshold)
        contours = []
        bitmap = (self.z > threshold).astype(int)
        self.visited = np.zeros(self.z.shape, dtype=bool)
        # check if contour start on edge (they have to touches the edge)
        for i in range(bitmap.shape[0] - 1):
            # left
            if self.corner_idx(bitmap[i:i+2, 0:2]) in [1, 3, 5, 7] and not self.visited[i, 0]:
                contours.append(self.find_contour_path(bitmap, i, 0))
            # right
            if self.corner_idx(bitmap[i:i+2, bitmap.shape[1]-2:bitmap.shape[1]]) in [4, 5, 12, 13] \
                    and not self.visited[i, bitmap.shape[1]-2]:
                contours.append(self.find_contour_path(bitmap, i, bitmap.shape[1]-2))

        for j in range(bitmap.shape[1] - 1):
            # top
            if self.corner_idx(bitmap[0:2, j:j+2]) in [8, 9, 10, 11] and not self.visited[0, j]:
                contours.append(self.find_contour_path(bitmap, 0, j))
            # bottom
            if self.corner_idx(bitmap[bitmap.shape[0]-2:bitmap.shape[0], j:j+2]) in [2, 6, 10, 14] \
                    and not self.visited[bitmap.shape[0]-2, j]:
                contours.append(self.find_contour_path(bitmap, bitmap.shape[0]-2, j))

        for i in range(bitmap.shape[0] - 1):
            for j in range(bitmap.shape[1] - 1):
                if self.corner_idx(bitmap[i:i+2, j:j+2]) not in [0, 15] and not self.visited[i, j]:
                    contours.append(self.find_contour_path(bitmap, i, j))
        return contours

    def find_contour_path(self, bitmap, start_i, start_j):
        i, j = start_i, start_j
        path = [self.to_real_coordinate(self.start_point(bitmap[i:i+2, j:j+2], np.array([i, j])).tolist())]

        previous_position = None
        while 0 <= i < bitmap.shape[0] - 1 \
                and 0 <= j < bitmap.shape[1] - 1\
                and not self.visited[i, j]:  # if visited true then cycle
            new_p = (self.new_point(bitmap[i:i+2, j:j+2], np.array(previous_position), np.array([i, j]))).tolist()
            path.append(self.to_real_coordinate(new_p))

            previous_position = [i, j]
            self.visited[i, j] = True
            i, j = self.new_position(bitmap[i:i+2, j:j+2], np.array(previous_position), np.array([i, j])).tolist()
        return path

    def to_real_coordinate(self, point):
        x_idx = int(point[1])
        y_idx = int(point[0])
        return [self.x[y_idx, x_idx] + ((point[1] % 1) * (self.x[y_idx, x_idx + 1] - self.x[y_idx, x_idx])
                if x_idx + 1 < self.x.shape[1] else 0),
                self.y[y_idx, x_idx] + ((point[0] % 1) * (self.y[y_idx + 1, x_idx] - self.y[y_idx, x_idx])
                if y_idx + 1 < self.x.shape[0] else 0)]

    @classmethod
    def new_point(cls, sq, previous, position):
        con_idx = cls.corner_idx(sq)
        if con_idx == 5:
            if previous is None:
                return position + np.array([0 if position[1] == 0 else 1, 0.5])  # on left edge 0 every time, same right
            return position + np.array([(0 if previous[1] + 1 == position[1] else 1), 0.5])
        elif con_idx == 10:
            if previous is None:
                return position + np.array([0.5, 1 if position[0] == 0 else 0])  # on top edge 1 every time, same bottom
            return position + np.array([0.5, (1 if previous is None or previous[0] + 1 == position[0] else 0)])
        else:
            return position + np.array(cls.corners[con_idx]['to'])

    @classmethod
    def start_point(cls, sq, position):
        con_idx = cls.corner_idx(sq)
        if con_idx == 5:
            return position + np.array([0.5 , 0 if position[1] == 0 else 1])  # on left edge 0 every time, same right
        elif con_idx == 10:
            return position + np.array([0 if position[0] == 0 else 1, 0.5])  # on top edge 1 every time, same bottom
        else:
            return position + np.array(cls.corners[con_idx]['from'])

    @classmethod
    def new_position(cls, sq, previous, position):
        con_idx = cls.corner_idx(sq)
        if con_idx == 5:
            return position + np.array([(-1 if previous is None or previous[1] + 1 == position[1] else 1), 0])
        elif con_idx == 10:
            return position + np.array([0, (1 if previous is None or previous[0] + 1 == position[0] else -1)])
        else:
            return position - (np.array(cls.corners[con_idx]['to']) == 0).astype(int)\
                   + (np.array(cls.corners[con_idx]['to']) == 1).astype(int)
            # move in position up/left if 0 in array or right/down if 1

    @staticmethod
    def corner_idx(sq):
        return np.sum(np.array([[8, 4], [1, 2]]) * sq)
