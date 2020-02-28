import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import copy

shape = (8, 16)
# MxNx2; first MxN is sequencer layer, second is 'active'/'playing field' layer.
# This describes state, *not* appearence.
grid = np.zeros(shape + (2,), dtype=np.int)
# Each cursor consists of a row, a height, a speed, and a position (column). ... and merge direction
# The first three things are fixed-ish, the last changes all the time.
# In particular, the last is a float which is rounded down for rendering and checking.
# TODO make it a claaaaass
# Idea: color spectrum across y axis; actual cursor color is average of colors in its rows.
# equivalently take average of start, start + height, compute color. okay yeah let's do that.
# another idea: separate merge and destroy effectors. destroy good for reducing complexity in a more serious way.
# alternatively, mute/unmute effector, allowing for same thing in a less permanent way.
class Cursor:
    def __init__(self, start, height, speed, pos, merge_direction):
        self.start = start
        self.height = height
        self.speed = speed
        self.pos = pos
        self.merge_direction = merge_direction

    def __repr__(self):
        return f"Cursor(start={self.start}, height={self.height}, speed={self.speed}, pos={self.pos}, merge_direction={self.merge_direction})"

proto_cursor = Cursor(0, 8, 3, 0, 1)
cursors = [copy.copy(proto_cursor)]


class Effector:
    def __init__(self, name, function, color):
        self.name = name
        self.function = function
        self.color = color


def reverse(cursor, _):
    if cursor.speed > 0:
        gap = cursor.pos - np.floor(cursor.pos)
        cursor.pos = np.floor(cursor.pos) + (1 - gap)
    else:
        gap = np.ceil(cursor.pos) - cursor.pos
        cursor.pos = np.ceil(cursor.pos) - (1 - gap)
    cursor.speed *= -1


def split(cursor, pos):
    if pos[0] == cursor.start:
        return  # can't split at the top. (think about it)
    global cursors
    ind = cursors.index(cursor)
    top = Cursor(cursor.start, pos[0] - cursor.start, cursor.speed, cursor.pos, 1)
    bottom = Cursor(pos[0], cursor.start + cursor.height - pos[0], cursor.speed, cursor.pos, -1)
    cursors[ind] = top
    cursors.insert(ind + 1, bottom)


def merge(cursor, _):
    ind = cursors.index(cursor)
    print("old", cursors)
    print("merge", ind, cursor.merge_direction)
    if cursor.merge_direction == -1:
        if ind > 0:
            print("with", ind - 1)
            cursors[ind - 1].height += cursor.height
        print("delete")
        del cursors[ind]
    elif cursor.merge_direction == 1:
        # Tricky issue: newly extended cursor will be processed after this one, and may register a hit on the same merge node!
        # How to avoid double-merging? One easy answer is that merge nodes disappear after use, but this is unsatisfactory.
        # Given how our system works, only one cursor should be on a row at a time - for it should never be the case that two cursors
        # both hit the same effector in one round. In which case, maybe the easiest thing to do is avoid activating any node twice.
        if ind < len(cursors) - 1:
            print("with", ind + 1)
            cursors[ind + 1].start = cursor.start
            cursors[ind + 1].height += cursor.height
        print("delete")
        del cursors[ind]
    print("new", cursors)


effectors = [
    Effector("reverse", reverse, (0, 1, 0)),
    Effector("split", split, (0, 0, 1)),
    Effector("merge", merge, (0, 0, 1)),
]
selected_effector = effectors[0]


def render(grid, cursors):
    # Render the state grid to an RGB image.
    g = np.ones(grid.shape[:2] + (3,), dtype=np.float) * [0.6, 0.6, 0.6]
    for cursor in cursors:
        middle = (cursor.start + (cursor.start + cursor.height)) / 2
        level = (middle - 0.5) / (grid.shape[0] - 1)
        g[cursor.start : cursor.start + cursor.height, int(cursor.pos)] = [level, 0, 1 - level]
    for r, c, layer in zip(*grid.nonzero()):
        value = grid[r, c, layer]
        g[r, c] = effectors[value - 1].color
    return g


matplotlib.rcParams["toolbar"] = "None"
plt.ion()
fig, ax = plt.subplots()


def on_click(event):
    if event.xdata is None or event.ydata is None:
        return
    x = int(round(event.xdata))
    y = int(round(event.ydata))
    print(x, y)
    grid[y, x] = effectors.index(selected_effector) + 1
    # g = render(grid, cursors)
    # im.set_data(g)


keymap = {
    "r": effectors[0],
    "s": effectors[1],
    # 'w': 'warp',
    "m": effectors[2],  # or 'join'?
}


def on_keypress(event):
    if event.key == "R":
        # Reset
        global cursors
        cursors = [copy.copy(proto_cursor)]
        return
    if event.key not in keymap:
        return
    global selected_effector
    selected_effector = keymap[event.key]
    ax.set_xlabel(selected_effector.name)


cid = fig.canvas.mpl_connect("button_press_event", on_click)
cid = fig.canvas.mpl_connect("key_press_event", on_keypress)

ax.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
    right=False,
    left=False,
    labelleft=False,
)
ax.set_xticks(np.arange(grid.shape[1] + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(grid.shape[0] + 1) - 0.5, minor=True)
ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)
for edge, spine in ax.spines.items():
    spine.set_visible(False)
im = ax.imshow(grid[:, :, 0])

last = time.time()


def update():
    global last
    now = time.time()
    dt = now - last
    visited = set()
    for cursor in cursors[:]:
        old_pos = int(cursor.pos)
        cursor.pos += cursor.speed * dt
        cursor.pos %= grid.shape[1]
        new_pos = int(cursor.pos)
        if new_pos != old_pos:
            hits = grid[cursor.start : cursor.start + cursor.height, new_pos, 0].nonzero()[0]
            for hit in hits:
                pos = (cursor.start + hit, new_pos)
                if pos in visited:
                    print(
                        f"aha, we already hit this effector ({pos}) this round; skipping for avoid merge issue."
                    )
                    continue
                visited.add(pos)
                effector = effectors[grid[pos[0], pos[1], 0] - 1]
                print(f"we hit {effector.name} at {pos}!")
                effector.function(cursor, pos)
    last = now


while True:
    g = render(grid, cursors)
    im.set_data(g)
    update()
    plt.pause(0.001)
