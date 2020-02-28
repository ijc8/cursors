import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

shape = (8, 8)
# MxNx2; first MxN is sequencer layer, second is 'active'/'playing field' layer.
# This describes state, *not* appearence.
grid = np.zeros(shape + (2,), dtype=np.int)
# Each cursor consists of a row, a height, a speed, and a position (column).
# The first three things are fixed-ish, the last changes all the time.
# In particular, the last is a float which is rounded down for rendering and checking.
# TODO class
cursors = [[0, 8, 1, 0]]

CURSOR_COLOR = [1, 0, 0]

class Effector:
    def __init__(self, name, function, color):
        self.name = name
        self.function = function
        self.color = color

def reverse(cursor):
    if cursor[2] > 0:
        gap = cursor[3] - np.floor(cursor[3])
        cursor[3] = np.floor(cursor[3]) + (1 - gap)
    else:
        gap = np.ceil(cursor[3]) - cursor[3]
        cursor[3] = np.ceil(cursor[3]) - (1 - gap)
    cursor[2] *= -1

def split(cursor):
    print('TODO')

effectors = [Effector('reverse', reverse, (0, 1, 0)),
             Effector('split', split, (0, 0, 1))]
selected_effector = effectors[0]

def render(grid, cursors):
    # Render the state grid to an RGB image.
    g = np.ones(grid.shape[:2] + (3,), dtype=np.float) * [0.6, 0.6, 0.6]
    for (start, height, _, pos) in cursors:
        g[start:start+height, int(pos)] = CURSOR_COLOR
    for r, c, layer in zip(*grid.nonzero()):
        value = grid[r, c, layer]
        g[r, c] = effectors[value - 1].color
    return g

matplotlib.rcParams['toolbar'] = 'None'
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
    'r': effectors[0],
    's': effectors[1],
    # 'w': 'warp',
    # 'm': 'merge',  # or 'join'?
}

def on_keypress(event):
    if event.key not in keymap:
        return
    global selected_effector
    selected_effector = keymap[event.key]
    ax.set_xlabel(selected_effector.name)

cid = fig.canvas.mpl_connect('button_press_event', on_click)
cid = fig.canvas.mpl_connect('key_press_event', on_keypress)

ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
ax.set_xticks(np.arange(grid.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(grid.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)
for edge, spine in ax.spines.items():
    spine.set_visible(False)
im = ax.imshow(grid[:, :, 0])

last = time.time()

def update():
    global last
    now = time.time()
    dt = now - last
    for cursor in cursors:
        old_pos = int(cursor[3])
        cursor[3] += cursor[2] * dt
        cursor[3] %= grid.shape[1]
        new_pos = int(cursor[3])
        start, height = cursor[:2]
        if new_pos != old_pos:
            hits = grid[start:start + height, new_pos, 0].nonzero()[0]
            for hit in hits:
                effector = effectors[grid[start + hit, new_pos, 0] - 1]
                print(f'we hit {effector.name} at ({start + hit}, {new_pos})!')
                effector.function(cursor)
    last = now

while True:
    g = render(grid, cursors)
    im.set_data(g)
    im.autoscale()
    update()
    plt.pause(0.001)
