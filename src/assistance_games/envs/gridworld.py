import copy

import assistance_games.rendering as rendering
from assistance_games.utils import get_asset

class Gridworld(object):
    """A gridworld with walls and multiple movable objects.

    Objects can move around using the four cardinal directions. They
    cannot move onto walls. Solid objects cannot occupy the same location as
    any other objects, while other objects can occupy the same locations as each
    other. The walls may be of different types: this does not affect the
    dynamics of the gridworld, but can be used externally.

    When used in environments, players would be represented as objects. In
    general, objects can be any hashable type.
    """
    def __init__(self, layout, object_positions, image_fns=None, solid_objects=None):
        """Sets up the gridworld.

        layout: Sequence of sequences of strings. If layout[y][x] == " ", that
            space is empty, otherwise it is a wall with type layout[y][x].
        object_positions: Dictionary mapping objects to positions (x, y). These
            are the positions at which each of the N objects starts.
        image_fns: Dictionary mapping objects and wall types to functions that
            given a rendering Viewer, position, height, and width, renders those
            objects and wall types. It is not required that every object and
            wall type has a corresponding renderer. make_rendering_fn (below)
            can be used to construct functions for common use cases.
        """
        self.height = len(layout)
        self.width = len(layout[0])
        self.layout = layout
        self.num_objects = len(object_positions)
        self.starting_positions = copy.deepcopy(object_positions)
        self.image_fns = {} if image_fns is None else image_fns
        self.solid_objects = set([]) if solid_objects is None else solid_objects
        self.viewer = None
        self.reset()

    def reset(self):
        self.object_positions = copy.deepcopy(self.starting_positions)

    def set_object_positions(self, object_positions):
        self.object_positions = copy.deepcopy(object_positions)

    def get_layout_type(self, pos):
        x, y = pos
        return self.layout[y][x]

    def is_in_bounds(self, pos):
        x, y = pos
        return (0 <= x < self.width) and (0 <= y < self.height)

    def is_free_location(self, pos):
        x, y = pos
        return self.is_in_bounds(pos) and self.layout[y][x] == " "

    def get_free_locations(self):
        """Returns a list of all possible states that objects can be in.

        Note it is not guaranteed that the agent can reach all of these states.
        """
        coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        all_states = [(x, y) for x, y in coords if not self.layout[y][x]]
        return all_states

    def get_move_location(self, obj, direction, object_positions=None):
        if object_positions is None:
            object_positions = self.object_positions
        pos = object_positions[obj]
        return Direction.move_in_direction(pos, direction)

    def functional_move(self, obj, direction, object_positions=None):
        """Moves obj in direction if possible, otherwise a noop."""
        if direction not in Direction.ALL_DIRECTIONS:
            raise ValueError("Illegal direction {}".format(direction))

        if object_positions is None:
            object_positions = self.object_positions

        pos = object_positions[obj]
        new_pos = Direction.move_in_direction(pos, direction)
        if new_pos == pos or not self.is_free_location(new_pos):
            return copy.deepcopy(object_positions)

        for other_obj, other_pos in object_positions.items():
            if other_obj == obj:
                continue
            if other_pos == new_pos and \
               (obj in self.solid_objects or other_obj in self.solid_objects):
                return copy.deepcopy(object_positions)

        new_positions = copy.deepcopy(object_positions)
        new_positions[obj] = new_pos
        return new_positions

    def move(self, obj, direction):
        self.object_positions = self.functional_move(obj, direction)

    def render(self, mode='human', finalized=False):
        # TODO: Currently this makes everything squarish
        h, w = self.height, self.width
        cell_shape = (200.0 / w, 200.0 / h)
        if self.viewer is None:
            import assistance_games.rendering as rendering
            self.viewer = rendering.Viewer(500, 600)
            self.viewer.set_bounds(-120, 120, -150, 120)

            self.grid = rendering.Grid(start=(-100, -100), end=(100, 100), shape=(w, h))
            self.viewer.add_geom(self.grid)

            for y in range(self.height):
                for x in range(self.width):
                    wall_type = self.layout[y][x]
                    if wall_type in self.image_fns:
                        self.image_fns[wall_type](self.viewer, self.grid, (x, y), cell_shape)

        for obj, pos in self.object_positions.items():
            if obj in self.image_fns:
                self.image_fns[obj](self.viewer, self.grid, pos, cell_shape)

        if finalized:
            return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def make_rendering_fn(asset_name, scale=None, offset=None, rgb_color=None):
    if scale is None: scale = 1
    if offset is None: offset = (0, 0)
    image, transform = None, None

    def fn(viewer, grid, pos, cell_shape):
        nonlocal image
        nonlocal transform
        cell_w, cell_h = cell_shape
        if image is None:
            filename = get_asset(asset_name)
            image = rendering.Image(filename, cell_w * scale, cell_h * scale)
            transform = rendering.Transform()
            image.add_attr(transform)
            if rgb_color is not None:
                image.set_color(*rgb_color)
            viewer.add_geom(image)

        x, y = pos
        offset_x, offset_y = offset
        x, y = x + offset_x, y + offset_y
        coords = grid.coords_from_pos((x, y))
        transform.set_translation(*coords)

    return fn
        

class Direction(object):
    """A class that contains the five actions available in Gridworlds.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST  = (1, 0)
    WEST  = (-1, 0)
    STAY = (0, 0)
    INDEX_TO_DIRECTION = [NORTH, SOUTH, EAST, WEST, STAY]
    DIRECTION_TO_INDEX = { a:i for i, a in enumerate(INDEX_TO_DIRECTION) }
    ALL_DIRECTIONS = INDEX_TO_DIRECTION

    @staticmethod
    def move_in_direction(point, direction):
        """Takes a step in the given direction and returns the new point.

        point: Tuple (x, y) representing a point in the x-y plane.
        direction: One of the Directions, except not Direction.STAY or
                   Direction.SELF_LOOP.
        """
        x, y = point
        dx, dy = direction
        return (x + dx, y + dy)

    @staticmethod
    def get_number_from_direction(direction):
        return Direction.DIRECTION_TO_INDEX[direction]

    @staticmethod
    def get_direction_from_number(number):
        return Direction.INDEX_TO_DIRECTION[number]
