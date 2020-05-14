import socket
import numpy as np
import select
import sys
import cursors
import json
import time
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import queue
import threading

# TODO: Consolidate with LP client.
def encode_bytes(window, pos, effector):
    first = window | (pos[0] << 5)
    second = pos[1] | (effector << 3)
    return bytes([first, second])

class ClientWindow:
    keymap = {
        "n": ("note", 0),
        "r": ("reverse", 1),
        "s": ("split", 2),
        "m": ("merge", 3),
        "w": ("warp", 4),
        "u": ("speedup", 5),
        "d": ("slowdown", 6),
        "0": ("erase", 7),
    }

    def __init__(self, player_id, state, framerate):
        self.player_id = player_id
        self.state = state
        self.running = True
        self.button_queue = queue.Queue()

        matplotlib.rcParams["toolbar"] = "None"
        fig, ax = plt.subplots(figsize=(7, 4))
        self.fig, self.ax = fig, ax
        bgcolor = '#29282b'
        fig.patch.set_facecolor(bgcolor)
        ax.set_facecolor(bgcolor)
        plt.tight_layout()

        fig.canvas.mpl_connect("close_event", self.on_close)
        fig.canvas.mpl_connect("button_press_event", self.on_click)
        fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        fig.canvas.mpl_connect("key_press_event", self.on_keypress)

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
        ax.set_xticks(np.arange(8 + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(8 + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=bgcolor, linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        im = ax.imshow(state.grid[:, (player_id*8):((player_id+1)*8)])
        self.im = im

        colors = [effector.rgb_color for effector in cursors.effectors] + [[1, 1, 1]]
        legend_lines = [matplotlib.lines.Line2D([0], [0], color=c, lw=8) for c in colors]
        legend_names = [f'{value[0]} ({key})' for key, value in self.keymap.items()]
        legend = fig.legend(legend_lines, legend_names, loc='upper left', frameon=False)
        plt.setp(legend.get_texts(), color='w')

        lc = ax.vlines([], [], [], color=state.cursors[0].rgb_color, lw=4)
        self.lc = lc
        self.ani = matplotlib.animation.FuncAnimation(fig, self.update_display, interval=1000/framerate)
        ax.xaxis.label.set_color('white')

    def run(self):
        plt.show(block=True)

    def on_close(self, event):
        self.running = False
        self.fig.canvas.stop_event_loop()

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        self.last_motion = (x, y)
        self.button_queue.put(((x, y), True))
        self.button_queue.put(((x, y), False))

    def on_motion(self, event):
        if event.button is None or event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if (x, y) == self.last_motion:
            return
        self.last_motion = (x, y)
        self.button_queue.put(((x, y), True))
        self.button_queue.put(((x, y), False))

    def on_keypress(self, event):
        if event.key not in self.keymap:
            return
        name, effector = self.keymap[event.key]
        self.ax.set_xlabel(name)
        self.button_queue.put(((8, effector), True))
        self.button_queue.put(((8, effector), False))

    def get_button_event(self):
        try:
            return self.button_queue.get(block=False)
        except queue.Empty:
            return None

    def render(self):
        # Render the state to an RGB image.
        g = np.ones((8, 8, 3), dtype=np.float) * [0.6, 0.6, 0.6]
        for r, c in zip(*self.state.grid[:, (self.player_id*8):((self.player_id+1)*8)].nonzero()):
            value = self.state.grid[r, c + self.player_id*8]
            g[r, c] = cursors.effectors[value - 1].rgb_color
        return g

    def update_display(self, frame):
        self.im.set_data(self.render())
        lines = np.array([[[c.pos - self.player_id*8, c.start - 0.5],
                           [c.pos - self.player_id*8, c.start + c.height]] for c in self.state.cursors])
        self.lc.set_segments(lines)


class CursorClient:
    def __init__(self, nickname=''):
        self.player_id = None
        self.nickname = nickname[:8]
        self.mirror_state = None
        self.modifiers = [False] * 8
        self.selected_effector = 1
        self.client_timestamp = None
        self.server_timestamp = None
        # TODO: decrease max frac_pos/time_offset on server side, and lower this.
        self.packet_jitter_buffer = 0.25  # 250 ms
        # Queue of incoming cursor changes, as list of (timestamp, [cursors]).
        self.cursor_updates = []
        self.cursor_timestamp = 0

    def open(self, host):
        self.frame = np.zeros((9, 8, 2))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, 8765))
        num_players, self.player_id = self.socket.recv(2)
        if self.player_id >= num_players:
            exit('Server full.')
        if not self.nickname:
            self.nickname = str(self.player_id)
        print(f'You are player {self.player_id} with nickname "{self.nickname}".')
        self.socket.send(bytes([len(self.nickname)]))
        self.socket.send(self.nickname.encode('utf8'))

        self.mirror_state = cursors.GameState(num_players)
        self.mirror_state.cursors[0].recv_pos = 0  # hack
        self.window = ClientWindow(self.player_id, self.mirror_state, 30)

        self.sockf = self.socket.makefile('r')

    def close(self):
        self.socket.close()

    def update_frame(self):
        if self.client_timestamp is None:
            return
        t = time.time()
        # Update to "current" set of cursors.
        updated = False
        while self.cursor_updates and t > self.cursor_updates[0][0]:
            self.cursor_timestamp, self.mirror_state.cursors = self.cursor_updates.pop(0)
            updated = True
        if updated:
            for cursor in self.mirror_state.cursors:
                cursor.recv_pos = cursor.pos
        # Interpolate cursor positions between updates
        for cursor in self.mirror_state.cursors:
            cursor.pos = (cursor.recv_pos + cursor.speed * (t - self.cursor_timestamp)) % self.mirror_state.grid.shape[1]

    def handle_input(self, event):
        pos = event[0]
        if pos[0] == 8:
            print(f'Column button {pos[1]} {event[1]}')
            self.modifiers[pos[1]] = event[1]
            if event[1]:
                self.selected_effector = pos[1] + 1
                if self.selected_effector > len(cursors.effectors):
                    self.selected_effector = 0
        elif event[1]:
            effector = self.selected_effector
            # Behave as a toggle, if the selected effector is already present at that location.
            global_pos = (pos[0] + self.player_id * 8, pos[1])
            if self.mirror_state.grid[global_pos[::-1]] == self.selected_effector:
                effector = 0
            data = encode_bytes(self.player_id, pos, effector)
            self.socket.send(data)

    def poll_server(self):
        while True:
            rs, _, _ = select.select([self.socket], [], [], 0)
            if not rs:
                break
            # TODO: optimize over-the-wire format as necessary
            data = json.loads(self.sockf.readline())
            timestamp = data['timestamp']
            now = time.time()
            if self.server_timestamp is None:
                self.client_timestamp = now + self.packet_jitter_buffer
                self.server_timestamp = timestamp
            self.client_timestamp += timestamp - self.server_timestamp
            self.server_timestamp = timestamp
            # print('server', timestamp, self.server_timestamp, timestamp - self.server_timestamp)
            delay = self.client_timestamp - now
            # print('client', old, self.client_timestamp, now, delay)
            if delay < 0:
                print(f'late by {-delay}. jitter exceeded {self.packet_jitter_buffer} seconds')
                delay = 0

            self.cursor_updates.append((self.client_timestamp, [cursors.Cursor(*d) for d in data['cursors']]))
            if 'grid' in data:
                self.mirror_state.grid = np.zeros(
                    self.mirror_state.grid.shape, dtype=np.int)
                for x, y, value in data.get('grid', []):
                    self.mirror_state.grid[x, y] = value

    def run(self):
        def run_state():
            while self.window.running:
                self.poll_server()
                event = self.window.get_button_event()
                if event:
                    self.handle_input(event)

                self.update_frame()
                time.sleep(1/60)

        t = threading.Thread(target=run_state)
        t.start()
        self.window.run()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit('usage: gui_client.py <server address> [nickname]')
    c = CursorClient(sys.argv[2] if len(sys.argv) > 2 else '')
    c.open(sys.argv[1])
    try:
        c.run()
    except KeyboardInterrupt:
        pass
    finally:
        c.close()
