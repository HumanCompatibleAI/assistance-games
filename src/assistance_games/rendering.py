"""Rendering support for environments.

Right now it is a simple wrapper around gym.envs.classic_control.rendering,
but might be expanded/replaced as needed.
"""

import numpy as np
import pyglet.text
from gym.envs.classic_control import rendering
from gym.envs.classic_control.rendering import (
    Geom, make_circle, Viewer, Transform, FilledPolygon, make_polygon)


def make_ellipse(r_x=10, r_y=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * np.pi * i / res
        points.append((np.cos(ang) * r_x, np.sin(ang) * r_y))
    return make_polygon(points, filled=filled)


class Image(rendering.Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_color(1.0, 1.0, 1.0)


class Grid(rendering.Geom):
    def __init__(self, *, start=(0, 0), end=None, grid_side=None, shape=(2, 2), invert_y=False):
        super().__init__()

        x_start, y_start = start
        self.cols, self.rows = shape

        if end is not None:
            x_end, y_end = end
            if invert_y:
                y_end, y_start = y_start, y_end
        else:
            x_end = x_start + self.cols * grid_side
            if invert_y:
                y_end = y_start - self.rows * grid_side
            else:
                y_end = y_start + self.rows * grid_side

        self.x_step = (x_end - x_start) / self.cols
        self.y_step = (y_end - y_start) / self.rows
        self.side = min(abs(self.x_step), abs(self.y_step))

        self.x0 = x_start + self.x_step / 2
        self.y0 = y_start + self.y_step / 2

        x_ticks = np.linspace(x_start, x_end, self.cols + 1)
        y_ticks = np.linspace(y_start, y_end, self.rows + 1)

        self.lines = []

        for x in x_ticks:
            line = rendering.Line((x, y_start), (x, y_end))
            self.lines.append(line)

        for y in y_ticks:
            line = rendering.Line((x_start, y), (x_end, y))
            self.lines.append(line)

    def pos_from_state(self, state):
        j, i = state % self.cols, state // self.cols
        return j, i

    def coords_from_pos(self, pos):
        j, i = pos
        x = self.x0 + self.x_step * j
        y = self.y0 + self.y_step * i
        return x, y

    def coords_from_state(self, state):
        return self.coords_from_pos(self.pos_from_state(state))

    def render1(self):
        for line in self.lines:
            line.render()

    def set_color(self, r, g, b):
        for line in self.lines:
            line.set_color(r, g, b)


class Text(rendering.Geom):
    def __init__(self, *, x=0, y=0, text='[placeholder]'):
        super().__init__()
        self.text = text
        self.label = pyglet.text.Label(self.text,
                                       font_name='Times New Roman',
                                       font_size=16,
                                       x=x, y=y,
                                       anchor_x='center', anchor_y='center',
                                       color=(0, 0, 0, 255))

    def render1(self):
        self.label.draw()
