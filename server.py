#!/usr/bin/env python3
import json
import socketserver
import sys
import threading
import time

import numpy as np
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

import cursors


class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print(f"-> {self.client_address[0]} connected")
        self.request.send(bytes([state.num_squares, len(clients)]))
        if len(clients) >= state.num_squares:
            print(f"no room, closing connection")
            return
        clients.add(self.request)

        try:
            while True:
                # self.request is the TCP socket connected to the client
                data = self.request.recv(2)
                if not data:
                    break
                assert(len(data) == 2)
                print("{} wrote:".format(self.client_address[0]))
                (x, y), effector = decode_bytes(data)
                state.grid[y, x] = effector
        except ConnectionResetError:
            pass

        clients.remove(self.request)
        print(f"<- {self.client_address[0]} disconnected")


# TODO: could just select() instead.
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        return json.JSONEncoder.default(self, obj)


def decode_bytes(b):
    first, second = b
    window = first & 0b11111
    pos = ((first >> 5) + (window * 8), second & 0b111)
    effector = second >> 3
    return (pos, effector)


def run():
    ### PLT stuff ###
    selected_effector = cursors.effectors[0]

    matplotlib.rcParams["toolbar"] = "None"
    fig, ax = plt.subplots(figsize=tuple(s // 2 for s in state.grid.shape[::-1]))
    running = True

    def on_close(event):
        nonlocal running
        running = False
        fig.canvas.stop_event_loop()

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        state.grid[y, x] = cursors.effectors.index(selected_effector) + 1

    keymap = {
        "n": cursors.effectors[0],
        "r": cursors.effectors[1],
        "s": cursors.effectors[2],
        "m": cursors.effectors[3],
        "w": cursors.effectors[4],
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

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_keypress)

    def plt_render(state):
        # Render the state to an RGB image.
        g = np.ones(state.grid.shape + (3,), dtype=np.float) * [0.6, 0.6, 0.6]
        for cursor in state.cursors:
            g[cursor.start : cursor.start + cursor.height, int(cursor.pos)] = cursor.rgb_color
        for r, c in zip(*state.grid.nonzero()):
            value = state.grid[r, c]
            g[r, c] = cursors.effectors[value - 1].rgb_color
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
    im = ax.imshow(state.grid[:, :])
    ### END PLT STUFF

    last = time.time()

    def update(frame):
        nonlocal last
        now = time.time()
        events = state.update(now - last)
        last = now

        g = plt_render(state)
        im.set_data(g)

        points = np.array(state.grid.nonzero()).T
        grid_info = []
        for p in points:
            grid_info.append([*p, state.grid[(*p,)]])
        data = {"cursors": [c.dump() for c in state.cursors]}
        if grid_info:
            data["grid"] = grid_info
        if events:
            data["events"] = events
        if data:
            data["timestamp"] = now
            data = json.dumps(data, cls=NumpyEncoder)
            for client in clients:
                try:
                    client.send(bytes(data + "\n", "utf8"))
                except OSError:
                    pass

    ani = matplotlib.animation.FuncAnimation(fig, update)

    try:
        fig.show()
        # TODO: is there a less ridiculous way to ensure that the GUI event loop is really in this thread?
        fig.canvas.stop_event_loop()
        fig.canvas.start_event_loop()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('usage: server.py <number of players>')
    state = cursors.GameState(int(sys.argv[1]))
    clients = set()
    with ThreadedTCPServer(("0.0.0.0", 8765), MyTCPHandler) as server:
        serve_thread = threading.Thread(target=server.serve_forever)
        serve_thread.start()
        run()
        server.shutdown()
