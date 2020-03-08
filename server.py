#!/usr/bin/env python3
import socketserver
import cursors
import threading
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

clients = set()
state = cursors.GameState()

class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        clients.add(self.request)
        while True:
            # self.request is the TCP socket connected to the client
            data = self.request.recv(1)
            if not data:
                break
            print("{} wrote:".format(self.client_address[0]))
            (x, y), color = decode_byte(data[0])
            state.grid[y, x] = 1
            for client in clients:
                if client is not self.request:
                    client.send(data)


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

ThreadedTCPServer.allow_reuse_addr = True


# TODO: support all 9 colors, instead of just 4.
def encode_byte(pos, color):
    return pos[0] | (pos[1] << 3) | (int(color[0] // 3) << 6) | (int(color[1] // 3) << 7)

def decode_byte(c):
    pos = (c & 0b111, (c >> 3) & 0b111)
    color = (((c >> 6) & 1) * 3, ((c >> 7) & 1) * 3)
    return (pos, color)

def render(state):
    # Render the state for the Launchpad.
    g = np.zeros(state.grid.shape[:2] + (2,), dtype=np.int)
    for cursor in state.cursors:
        middle = (cursor.start + (cursor.start + cursor.height)) / 2
        level = (middle - 0.5) / (state.grid.shape[0] - 1)
        g[cursor.start : cursor.start + cursor.height, int(cursor.pos)] = [
            3, # round(level * 3),
            3, # 3 - round(level * 3),
        ]
    for r, c, layer in zip(*state.grid.nonzero()):
        value = state.grid[r, c, layer]
        # g[r, c] = effectors[value - 1].color
        g[r, c] = [0, 3]

    return g[:8, :8, :].swapaxes(0, 1)  # return first square

def run():
    ### PLT stuff ###
    selected_effector = cursors.effectors[0]

    matplotlib.rcParams["toolbar"] = "None"
    plt.ion()
    fig, ax = plt.subplots()

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        print(x, y)
        state.grid[y, x] = cursors.effectors.index(selected_effector) + 1
        # g = render(grid, cursors)
        # im.set_data(g)

    keymap = {
        "r": cursors.effectors[0],
        "s": cursors.effectors[1],
        # 'w': 'warp',
        "m": cursors.effectors[2],  # or 'join'?
    }

    def on_keypress(event):
        if event.key == "R":
            # Reset
            state.reset_cursors()
            return
        if event.key not in keymap:
            return
        nonlocal selected_effector
        selected_effector = keymap[event.key]
        ax.set_xlabel(selected_effector.name)

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    cid = fig.canvas.mpl_connect("key_press_event", on_keypress)

    def plt_render(state):
        # Render the state to an RGB image.
        g = np.ones(state.grid.shape[:2] + (3,), dtype=np.float) * [0.6, 0.6, 0.6]
        for cursor in state.cursors:
            middle = (cursor.start + (cursor.start + cursor.height)) / 2
            level = (middle - 0.5) / (state.grid.shape[0] - 1)
            g[cursor.start : cursor.start + cursor.height, int(cursor.pos)] = [
                level,
                0,
                1 - level,
            ]
        for r, c, layer in zip(*state.grid.nonzero()):
            value = state.grid[r, c, layer]
            g[r, c] = cursors.effectors[value - 1].color
        return g

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
    ax.set_xticks(np.arange(state.grid.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(state.grid.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    im = ax.imshow(state.grid[:, :, 0])
    ### END PLT STUFF

    last = time.time()
    last_frame = np.zeros((8, 8, 2))

    while True:
        now = time.time()
        state.update(last - now)
        last = now


        g = plt_render(state)
        im.set_data(g)

        frame = render(state)
        r = np.where(frame != last_frame)
        xs, ys, _ = r
        # Ignore duplicates.
        points = set(zip(xs, ys))
        data = []
        for x, y in points:
            color = frame[x, y]
            data.append(encode_byte((x, y), color))
        if data:
            # print(len(bytes(data)), bytes(data))
            for client in clients:
                client.send(bytes(data))
        last_frame = frame

        plt.pause(0.0001)

run_thread = threading.Thread(target=run)
run_thread.start()

with ThreadedTCPServer(('0.0.0.0', 8765), MyTCPHandler) as server:
    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    # server_thread = threading.Thread(target=server.serve_forever)
    # server_thread.start()
    server.serve_forever()
