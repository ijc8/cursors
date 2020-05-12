#!/usr/bin/env python3
import json
import socketserver
import sys
import threading
import time

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation
import matplotlib.pyplot as plt
from oscpy.client import OSCClient

import cursors

def get_unused_player_id():
    ids = {id for id, sock in clients}
    pool = set(range(len(ids) + 1))
    return sorted(pool - ids)[0]


class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print(f"-> {self.client_address[0]} connected")
        player_id = get_unused_player_id()
        self.request.send(bytes([state.num_squares, player_id]))
        if len(clients) >= state.num_squares:
            print(f"no room, closing connection")
            return
        nickname_len = self.request.recv(1)[0]
        nickname = self.request.recv(nickname_len).decode('utf8')
        nicknames[player_id] = nickname
        print('players', nicknames)
        clients.add((player_id, self.request))
        # Send the client initial state upon join, without first waiting for an event.
        now = time.time()
        points = np.array(state.grid.nonzero()).T
        grid_info = []
        for p in points:
            grid_info.append([*p, state.grid[(*p,)]])
        data = {}
        data["grid"] = grid_info
        data["cursors"] = [c.dump() for c in state.cursors]
        data["timestamp"] = now
        data = json.dumps(data, cls=NumpyEncoder)
        try:
            self.request.send(bytes(data + "\n", "utf8"))
        except OSError:
            pass

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

        clients.remove((player_id, self.request))
        nicknames[player_id] = ''
        print(f"<- {self.client_address[0]} disconnected")


# TODO: could just select() instead.
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def np_convert(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj.item()
    return obj

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        return np_convert(obj)


def decode_bytes(b):
    first, second = b
    window = first & 0b11111
    pos = ((first >> 5) + (window * 8), second & 0b111)
    effector = second >> 3
    return (pos, effector)


# TODO: avoid duplicating name/order
keymap = {
    "0": ("erase", 0),
    "n": ("note", 1),
    "r": ("reverse", 2),
    "s": ("split", 3),
    "m": ("merge", 4),
    "w": ("warp", 5),
    "u": ("speedup", 6),
    "d": ("slowdown", 7),
}


def run():
    osc_client = OSCClient('127.0.0.1', 8000)
    state_framerate = 60
    graphics_framerate = 30

    ### PLT stuff ###
    selected_effector = keymap["n"]

    matplotlib.rcParams["toolbar"] = "None"
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    plt.tight_layout()
    running = True
    hole_radius = 3
    # r = (np.arange(state.grid.shape[0] + 1) + hole_radius)[::-1]
    lo = 4
    r = (np.geomspace(lo, state.grid.shape[0] + lo, num=state.grid.shape[0] + 1) - lo + hole_radius)[::-1]
    print('rs', r - hole_radius)

    def on_close(event):
        nonlocal running
        running = False
        fig.canvas.stop_event_loop()

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        col = int(np.floor(-event.xdata / (2*np.pi) * state.grid.shape[1]))
        row = np.where(r > event.ydata)[0][-1]
        state.grid[row, col] = selected_effector[1]

    def on_keypress(event):
        if event.key == "R":
            # Reset
            state.reset_cursors()
            return
        if event.key not in keymap:
            return
        nonlocal selected_effector
        selected_effector = keymap[event.key]
        ax.set_xlabel(selected_effector[0])

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_keypress)

    def plt_render(state):
        # Render the state to an RGB image.
        g = np.ones(state.grid.shape + (3,), dtype=np.float) * [0.6, 0.6, 0.6]
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
    theta = np.linspace(0, 2*np.pi, state.grid.shape[1] + 1)

    image = plt_render(state)[:, ::-1, :]
    raveled_pixel_shape = (image.shape[0]*image.shape[1], image.shape[2])
    color_tuple = image.reshape(raveled_pixel_shape)

    index = np.tile(np.arange(image.shape[0]), (image.shape[1],1))
    im = ax.pcolormesh(theta, r, index.T, color=color_tuple, linewidth=0)
    im.set_array(None)

    bgcolor = '#29282b'
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor("black")
    # Setting xticks mysteriously misses some points.
    ax.vlines(theta, hole_radius, r[0], color=bgcolor, linewidth=3)
    # Also, show divisions between squares:
    ax.vlines(theta[::8], hole_radius, r[0], color='black', linewidth=5)
    ax.set_rticks(r, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(which="minor", color=bgcolor, linestyle="-", linewidth=3)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    lc = ax.vlines([], [], [], color=state.cursors[0].rgb_color, lw=4)

    nickname_texts = []
    for i, name in enumerate(nicknames):
        print(name)
        angle = -(i + .5) / num_players * 2*np.pi
        nickname_texts.append(ax.text(
            angle, hole_radius + 8.5, name,
            rotation=angle / np.pi * 180 - 90,
            va='center', ha='center',
            fontweight=10,
            fontfamily='monospace',
            fontsize=18,
            color='gray'
        ))

    def update_display(frame):
        image = plt_render(state)[:, ::-1, :]
        raveled_pixel_shape = (image.shape[0]*image.shape[1], image.shape[2])
        color_tuple = image.reshape(raveled_pixel_shape)
        im.set_color(color_tuple)
        lines = np.array([[[-c.pos / state.grid.shape[1] * 2 * np.pi, r[c.start]],
                           [-c.pos / state.grid.shape[1] * 2 * np.pi, r[c.start + c.height]]] for c in state.cursors])
        lc.set_segments(lines)
        for name, text in zip(nicknames, nickname_texts):
            text.set_text(name)


    ani = matplotlib.animation.FuncAnimation(fig, update_display, interval=1000/graphics_framerate)
    ### END PLT STUFF

    def update_state():
        last = time.time()
        last_grid_info = []
        while running:
            now = time.time()
            events = state.update(now - last)
            last = now

            points = np.array(state.grid.nonzero()).T
            grid_info = []
            for p in points:
                grid_info.append([*p, state.grid[(*p,)]])
            data = {}
            if grid_info != last_grid_info:
                data["grid"] = grid_info
                last_grid_info = grid_info
            if events:
                for event in events:
                    event[0] = event[0].encode('utf8')
                    event[1:] = map(np_convert, event[1:])
                    event[-1] += 1/state_framerate
                    osc_client.send_message(b'/cursors', event)
            if data or events:
                # Only send cursor updates if something else of interest has occurred.
                # Assumes the client will interpolate motion based on cursor speed in the meantime.
                data["cursors"] = [c.dump() for c in state.cursors]
                data["timestamp"] = now
                data = json.dumps(data, cls=NumpyEncoder)
                for id, sock in clients.copy():
                    try:
                        sock.send(bytes(data + "\n", "utf8"))
                    except OSError:
                        pass
            time.sleep(1/state_framerate)

    t = threading.Thread(target=update_state)
    t.start()

    # legend_lines = [matplotlib.lines.Line2D([0], [0], color=effector.rgb_color, lw=8) for effector in cursors.effectors]
    # legend_names = [effector.name for effector in cursors.effectors]
    # legend = fig.legend(legend_lines, legend_names, loc='upper left', frameon=False)
    # plt.setp(legend.get_texts(), color='w')

    plt.show(block=True)
    running = False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('usage: server.py <number of players>')
    num_players = int(sys.argv[1])
    state = cursors.GameState(num_players)
    clients = set()
    nicknames = [''] * num_players
    with ThreadedTCPServer(("0.0.0.0", 8765), MyTCPHandler) as server:
        serve_thread = threading.Thread(target=server.serve_forever)
        serve_thread.start()
        run()
        server.shutdown()
