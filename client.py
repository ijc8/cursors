import socket
import numpy as np
import my_launchpad as launchpad
import select
import sys
import cursors
import json


def from_grid(x, y):
    # Assume x and y are in range (0, 8).
    return x + y * 16

def to_grid(v):
    return (v % 16, v // 16)

def decode_byte(c):
    pos = (c & 0b111, (c >> 3) & 0b111)
    color = (((c >> 6) & 1) * 3, ((c >> 7) & 1) * 3)
    return (pos, color)

def encode_byte(pos, color):
    return pos[0] | (pos[1] << 3) | (int(color[0] // 3) << 6) | (int(color[1] // 3) << 7)

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
    for r, c in zip(*state.grid.nonzero()):
        value = state.grid[r, c]
        # g[r, c] = effectors[value - 1].color
        g[r, c] = [0, 3]

    return g[:8, :8, :].swapaxes(0, 1)  # return first square


# Thoughts on protocol:
# - update to cursor position
# - update to cursor list
# - change to board: remove f at (x, y), or add f at (x, y). more complicated with lifespan.
# 00XXXYYY: update cursor #XXX to have position YYY.
# 10XXXYYY: remove something at (x, y), to be specified in the next byte
# 11XXXYYY: add something at (x, y), to be specified in the next byte
# But for now, this is premature optimization! Let's just be verbose and use readline().


class LaunchpadException(Exception):
    pass


class CursorClient:
    def __init__(self):
        self.lp = launchpad.Launchpad()
        self.mirror_state = cursors.GameState()

    def open(self, host):
        if not self.lp.Open():
            raise LaunchpadException('No launchpad detected.')
        self.lp.ButtonFlush()
        self.frame = np.zeros((8, 8, 2))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, 8765))
        self.sockf = self.socket.makefile('r')

    def close(self):
        self.lp.Reset()
        self.lp.Close()

    def update_frame(self):
        self.next_frame = render(self.mirror_state)
        r = np.where(self.next_frame != self.frame)
        xs, ys, _ = r
        # Ignore duplicates.
        points = set(zip(xs, ys))
        for x, y in points:
            color = self.next_frame[x, y]
            self.lp.LedCtrlRaw(from_grid(x, y), *color)
        self.frame = self.next_frame

    def handle_input(self, event):
        if not event[1]:
            return
        pos = to_grid(event[0])
        data = bytes([encode_byte(pos, (3, 3))])
        self.socket.send(data)

    def poll_server(self):
        while True:
            rs, _, _ = select.select([self.socket], [], [], 0)
            if not rs:
                break
            #c = self.socket.recv(1)
            data = json.loads(self.sockf.readline())
            self.mirror_state.grid = np.zeros(self.mirror_state.grid.shape, dtype=np.int)
            self.mirror_state.cursors = [cursors.Cursor(*d) for d in data['cursors']]
            for x, y, value in data.get('grid', []):
                self.mirror_state.grid[x, y] = value
            for event in data.get('events', []):
                print(f'Event: {event}')
            # pos, color = decode_byte(c[0])
            # self.next_frame[pos] = color

    def run(self):
        while True:
            self.next_frame = self.frame.copy()
            self.poll_server()
            event = self.lp.ButtonStateRaw()
            if event != []:
                self.handle_input(event)

            self.update_frame()


if __name__ == '__main__':
    c = CursorClient()
    c.open(sys.argv[1])
    try:
        c.run()
    except KeyboardInterrupt:
        pass
    finally:
        c.close()
