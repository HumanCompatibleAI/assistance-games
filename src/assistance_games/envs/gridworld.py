import copy

import assistance_games.rendering as rendering
from ..pyglet_rendering import Transform, Viewer
from ..utils import get_asset

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
    def __init__(self, layout, object_positions, rendering_fns=None, solid_objects=None):
        """Sets up the gridworld.

        layout: Sequence of sequences of strings. If layout[y][x] == " ", that
            space is empty, otherwise it is a wall with type layout[y][x].
        object_positions: Dictionary mapping objects to positions (x, y). These
            are the positions at which each of the N objects starts.
        image_fns: Dictionary mapping objects and wall types to lists of
            functions that render those objects and wall types. It is not
            required that every object and wall type be present.
            make_rendering_fn (below) can be used to construct functions for
            common use cases.
        """
        self.height = len(layout)
        self.width = len(layout[0])
        self.layout = layout
        self.num_objects = len(object_positions)
        self.starting_positions = copy.deepcopy(object_positions)
        self.rendering_fns = {} if rendering_fns is None else rendering_fns
        self.solid_objects = set([]) if solid_objects is None else solid_objects
        self.viewer = None
        self.reset()

    def reset(self):
        self.object_positions = copy.deepcopy(self.starting_positions)
        self.object_orientations = {obj : Direction.NORTH for obj in self.object_positions.keys()}

    def set_object_positions(self, object_positions):
        self.object_positions = copy.deepcopy(object_positions)

    def set_object_orientations(self, object_orientations):
        self.object_orientations = copy.deepcopy(object_orientations)

    def get_layout_type(self, pos):
        x, y = pos
        return self.layout[y][x]

    def get_layout_positions(self, wall_type):
        result = []
        for y, row in enumerate(self.layout):
            for x, cell in enumerate(row):
                if cell == wall_type:
                    result.append((x, y))
        return result

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
        if object_positions is None: object_positions = self.object_positions
        pos = object_positions[obj]
        return Direction.move_in_direction(pos, direction)

    def get_facing_position(self, obj, object_positions=None, object_orientations=None):
        if object_positions is None: object_positions = self.object_positions
        if object_orientations is None: object_orientations = self.object_orientations

        x, y = object_positions[obj]
        dx, dy = object_orientations[obj]
        return (x + dx, y + dy)

    def get_facing_wall_type(self, obj, object_positions=None, object_orientations=None):
        x, y = self.get_facing_position(obj, object_positions, object_orientations)
        return self.layout[y][x]

    def functional_move(self, obj, direction, object_positions, object_orientations=None):
        """Moves obj in direction if possible, and returns the new state.

        Always returns the new object positions (making a copy, rather than
        modifying in place). If object_orientations is not None, also returns
        the new object orientations.
        """
        if direction not in Direction.ALL_DIRECTIONS:
            raise ValueError("Illegal direction {}".format(direction))

        new_positions = copy.deepcopy(object_positions)
        new_orientations = copy.deepcopy(object_orientations)
        if direction != Direction.STAY and new_orientations is not None:
            new_orientations[obj] = direction

        pos = object_positions[obj]
        new_pos = Direction.move_in_direction(pos, direction)
        if new_pos == pos or not self.is_free_location(new_pos):
            return new_positions, new_orientations

        for other_obj, other_pos in object_positions.items():
            if other_obj == obj:
                continue
            if other_pos == new_pos and \
               (obj in self.solid_objects or other_obj in self.solid_objects):
                return new_positions, new_orientations

        new_positions[obj] = new_pos
        return new_positions, new_orientations


    def move(self, obj, direction):
        """Moves obj in direction if possible, otherwise a noop."""
        self.object_positions, self.object_orientations = self.functional_move(
            obj, direction, self.object_positions, self.object_orientations)

    def is_renderer_initialized(self):
        return self.viewer is not None

    def initialize_renderer(self, viewer_bounds=(600, 600), grid_offsets=(0, 0, 0, 0)):
        assert not self.is_renderer_initialized()
        # TODO: Make the viewer grid adapt to the gridworld dimensions
        h, w = self.height, self.width
        cell_shape = (200.0 / w, 200.0 / h)

        import assistance_games.rendering as rendering
        self.viewer = Viewer(*viewer_bounds)
        bounds = (-120, 120, -120, 120)
        bounds = tuple((b + o for b, o in zip(bounds, grid_offsets)))
        self.viewer.set_bounds(*bounds)

        self.grid = rendering.Grid(start=(-100, -100), end=(100, 100), shape=(w, h))
        self.viewer.add_geom(self.grid)

        for y in range(self.height):
            for x in range(self.width):
                wall_type = self.layout[y][x]
                for fn in self.rendering_fns.get(wall_type, []):
                    fn()(self.viewer, self.grid, (x, y), cell_shape)

        self.object_renderers = {}
        for obj, pos in self.object_positions.items():
            self.object_renderers[obj] = []
            for fn in self.rendering_fns.get(obj, []):
                self.object_renderers[obj].append(fn())

    def render(self, mode='human', finalized=False):
        if not self.is_renderer_initialized():
            self.initialize_renderer()

        h, w = self.height, self.width
        cell_shape = (200.0 / w, 200.0 / h)
        for obj, pos in self.object_positions.items():
            for fn in self.object_renderers[obj]:
                fn(self.viewer, self.grid, pos, cell_shape)

        if finalized:
            return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


# TODO: This has become a fairly complicated interface, maybe switch to
# object-oriented style
def make_rendering_fn(creation_fn, offset=None, rgb_color=None):
    if offset is None: offset = (0, 0)

    def creator():
        transform = None
        def fn(viewer, grid, pos, cell_shape):
            nonlocal transform
            if transform is None:
                thing = creation_fn(viewer, grid, pos, cell_shape)
                transform = Transform()
                thing.add_attr(transform)
                if rgb_color is not None:
                    thing.set_color(*rgb_color)
                viewer.add_geom(thing)

            cell_w, cell_h = cell_shape
            x, y = pos
            offset_x, offset_y = offset
            x, y = x + offset_x, y + offset_y
            coords = grid.coords_from_pos((x, y))
            transform.set_translation(*coords)
        return fn
    return creator


def make_image_renderer(asset_name, scale=None, offset=None, rgb_color=None):
    if scale is None: scale = 1

    def creation_fn(viewer, grid, pos, cell_shape):
        w, h = cell_shape
        filename = get_asset(asset_name)
        return rendering.Image(filename, w * scale, h * scale)

    return make_rendering_fn(creation_fn, offset=offset, rgb_color=rgb_color)


def make_cell_renderer(rgb_color):
    def creation_fn(viewer, grid, pos, cell_shape):
        w, h = cell_shape
        x, y = w / 2, h / 2
        return rendering.make_polygon([(-x, -y), (x, -y),(x, y),(-x, y)])
    return make_rendering_fn(creation_fn, rgb_color=rgb_color)


def make_ellipse_renderer(scale_width=1, scale_height=None, offset=None, rgb_color=None):
    if scale_height is None: scale_height = scale_width
    def creation_fn(viewer, grid, pos, cell_shape):
        w, h = cell_shape
        return rendering.make_ellipse(scale_width * w/2, scale_height * h/2)
    return make_rendering_fn(creation_fn, offset=offset, rgb_color=rgb_color)


class Direction(object):
    """A class that contains the five actions available in Gridworlds.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """
    NORTH = (0, 1)
    SOUTH = (0, -1)
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

    @staticmethod
    def get_component_directions(composite_direction):
        dx, dy = composite_direction
        assert dx != 0 or dy != 0

        def sign(num):
            return 0 if num == 0 else (1 if num > 0 else -1)

        if dx == 0:
            return [(0, sign(dy))]
        elif dy == 0:
            return [(sign(dx), 0)]
        return [(sign(dx), 0), (0, sign(dy))]
