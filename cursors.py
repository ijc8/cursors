import numpy as np
import time
import copy


def lp_to_rgb(color):
    return (color[0] / 3, color[1] / 3, 0)


# Each cursor consists of a row, a height, a speed, and a position (column). ... and merge direction
# The first three things are fixed-ish, the last changes all the time.
# In particular, the last is a float which is rounded down for rendering and checking.

# idea: separate merge and destroy effectors. destroy good for reducing complexity in a more serious way.
# alternatively, mute/unmute effector, allowing for same thing in a less permanent way.
class Cursor:
    def __init__(self, start, height, speed, pos, merge_direction):
        self.start = start
        self.height = height
        self.speed = speed
        self.pos = pos
        self.merge_direction = merge_direction
        self.color = [1, 1]
        self.rgb_color = lp_to_rgb(self.color)

    def dump(self):
        return [self.start, self.height, self.speed, self.pos, self.merge_direction]

    def get_frac_pos(self):
        "Compute how far this cursor is past the nearest column division."
        return self.pos - np.floor(self.pos) if self.speed > 0 else np.ceil(self.pos) - self.pos

    def set_frac_pos(self, frac):
        self.pos = np.floor(self.pos) + frac if self.speed > 0 else np.ceil(self.pos) - frac

    def __repr__(self):
        return f"Cursor(start={self.start}, height={self.height}, speed={self.speed}, pos={self.pos}, merge_direction={self.merge_direction})"


proto_cursor = Cursor(0, 8, 4, 0, 1)


class GameState:
    def __init__(self, num_squares):
        """Setup the game state.

        num_squares is the number of consecutive square grids (probably equal to the number of players).
        """
        self.num_squares = num_squares
        self.cursors = [copy.copy(proto_cursor)]
        shape = (8, num_squares * 8)
        # MxN playing field of different effects & triggers.
        # TODO: Will each element be a bitmask of things at that location?
        # Seem like no, if effectors have their own parameters (lifespan, maybe destination for warps)
        # This describes state, *not* appearence.
        self.grid = np.zeros(shape, dtype=np.int)

    def reset_cursors(self):
        self.cursors = [copy.copy(proto_cursor)]

    def get_warps(self, start_row, end_row):
        "Get all the warps in the rows [start_row, end_row). Returns columns of valid warps."
        # TODO convenient way to go from effector name to id - avoid magic numbers.
        return list(np.unique(np.where(self.grid[start_row:end_row] == 5)[1]))

    def update(self, dt):
        queue = []
        # First, update cursor positions and add ones that entered a new column to the queue.
        for cursor in self.cursors:
            old_pos = int(cursor.pos)
            # NOTE: This does not handle the case where dt is so large that multiple steps have passed.
            # If dt is too large, cursors will 'teleport' to their new position, skipping any effectors in between.
            # This can dealt with higher up by making sure dt is never larger than the highest cursor speed.
            cursor.pos += cursor.speed * dt
            cursor.pos %= self.grid.shape[1]
            new_pos = int(cursor.pos)
            if new_pos != old_pos:
                queue.append(cursor)

        events = []
        visited = set()
        # Then run events. Note that effectors may cause the queue to grow while we're processing it.
        while queue:
            cursor = queue.pop()
            if cursor not in self.cursors:
                continue
            new_pos = int(cursor.pos)
            frac_pos = cursor.get_frac_pos()
            # Get the precise moment when this cursor hit this column, as an offset from 'now'.
            time_offset = -abs(frac_pos / cursor.speed)
            hits = self.grid[
                cursor.start : cursor.start + cursor.height, new_pos
            ].nonzero()[0]
            hits = [((cursor.start + hit, new_pos), effectors[self.grid[cursor.start + hit, new_pos] - 1]) for hit in hits]
            hits.sort(key=lambda p: p[1].order)
            for pos, effector in hits:
                if pos in visited:
                    # Already used this effector on an earlier cursor (pre-split)
                    continue
                visited.add(pos)
                print(f"we hit {effector.name} at {pos}!")
                events.append([effector.name, *pos, cursor.height, cursor.speed, time_offset])
                enqueued = effector.function(self, cursor, pos)
                if enqueued is not None:
                    # Effector manipulated the queue; stop processing this cursor.
                    queue += enqueued
                    break
        return events


class Effector:
    def __init__(self, name, function, color, order=0):
        # NOTE: color is a launchpad mk2 color (r, g), where r and g are in [0, 3].
        self.name = name
        self.function = function
        self.color = color
        self.rgb_color = lp_to_rgb(color)
        self.order = order


def reverse(state, cursor, _):
    cursor.set_frac_pos(1 - cursor.get_frac_pos())
    cursor.speed *= -1


def split(state, cursor, pos):
    if pos[0] == cursor.start:
        return  # can't split at the top. (think about it)
    ind = state.cursors.index(cursor)
    top = Cursor(cursor.start, pos[0] - cursor.start, cursor.speed, cursor.pos, 1)
    bottom = Cursor(pos[0], cursor.start + cursor.height - pos[0], cursor.speed, cursor.pos, -1)
    state.cursors[ind] = top
    state.cursors.insert(ind + 1, bottom)
    # Add children to queue.
    return [top, bottom]


def merge(state, cursor, _):
    ind = state.cursors.index(cursor)
    if ind > 0 and (cursor.merge_direction < 0 or ind == len(state.cursors) - 1):
        # Merge upwards
        state.cursors[ind - 1].height += cursor.height
        del state.cursors[ind]
        # Re-process parent in case it should hit any other merge effectors with its new height.
        # return [state.cursors[ind - 1]]
        return []
    elif ind < len(state.cursors) - 1 and (cursor.merge_direction > 0 or ind == 0):
        # Merge downwards
        state.cursors[ind + 1].start = cursor.start
        state.cursors[ind + 1].height += cursor.height
        del state.cursors[ind]
        # Re-process parent in case it should hit any other merge effectors with its new height.
        # return [state.cursors[ind]]
        return []


def warp(state, cursor, pos):
    warps = state.get_warps(cursor.start, cursor.start + cursor.height)
    dir = 1 if cursor.speed > 0 else -1
    # Add difference (rather than setting directly) to preserve fractional position:
    cursor.pos += warps[(warps.index(pos[1]) + dir) % len(warps)] - pos[1]


def speedup(state, cursor, pos):
    if abs(cursor.speed) < 16:
        cursor.speed *= 2
        cursor.set_frac_pos(cursor.get_frac_pos() * 2)
    state.grid[pos] = 0  # self-destruct


def slowdown(state, cursor, pos):
    if abs(cursor.speed) > 0.5:
        cursor.speed /= 2
        cursor.set_frac_pos(cursor.get_frac_pos() / 2)
    state.grid[pos] = 0  # self-destruct


def trigger(*_):
    pass  # doesn't modify game state


# TODO: reconsider colors and ordering
effectors = [
    Effector("note", trigger, (0, 1)),
    Effector("reverse", reverse, (3, 2)),
    # Note that split is applied after all effects (except merge)
    # This is to avoid bizarre behaviors and asymmetry that arise otherwise.
    # For example, if a cursor hits split and speedup at the same instant,
    # both of the child cursors get the speedup (as opposed to just one or neither).
    Effector("split", split, (3, 0), order=1),
    # Merge is applied after all other effects
    Effector("merge", merge, (1, 0), order=2),
    Effector("warp", warp, (2, 3)),
    Effector("speedup", speedup, (0, 3)),
    Effector("slowdown", slowdown, (1, 2)),
]
