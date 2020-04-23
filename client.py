import socket
import numpy as np
import launchpad
import select
import sys
import cursors
import json
import time
from oscpy.client import OSCClient


def encode_bytes(window, pos, effector):
    first = window | (pos[0] << 5)
    second = pos[1] | (effector << 3)
    return bytes([first, second])


def render(state, modifiers, start_col):
    # Render the state for the Launchpad.
    # Each player gets an 8-column window into the full game state.
    g = np.zeros((8, 9, 2), dtype=np.int)
    for cursor in state.cursors:
        # TODO adjust bounds for each player's client
        if start_col <= cursor.pos < start_col + 8:
            g[cursor.start: cursor.start + cursor.height, int(cursor.pos) - start_col] = cursor.color

    # Uncomment to examine the Launchpad's palette:
    # for r in range(4):
    #     for y in range(4):
    #         g[r, y] = [r, y]

    show_all = not any(modifiers[:len(cursors.effectors)])
    for r, c in zip(*state.grid.nonzero()):
        # TODO adjust bounds for each player's client
        if start_col <= c < start_col + 8:
            value = state.grid[r, c]
            if show_all or modifiers[value - 1]:
                g[r, c - start_col] += cursors.effectors[value - 1].color

    # Set column button colors to match the effectors they select.
    for i, effector in enumerate(cursors.effectors):
        if not modifiers[i]:
            # Feedback: when select button is pressed, light turns off.
            g[i, 8] = effector.color

    g = np.clip(g, 0, 3)

    return g.swapaxes(0, 1)


class CursorClient:
    def __init__(self, player_id):
        self.player_id = player_id
        self.lp = launchpad.Launchpad()
        self.mirror_state = cursors.GameState()
        self.modifiers = [False] * 8
        self.selected_effector = 1
        self.server_timestamp = None
        # TODO: decrease max frac_pos/time_offset on server side, and lower this.
        self.packet_jitter_buffer = 0.25  # 250 ms

    def open(self, host):
        self.lp.open()
        self.lp.flush_button_events()
        self.frame = np.zeros((9, 8, 2))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, 8765))
        self.sockf = self.socket.makefile('r')
        self.client = OSCClient('127.0.0.1', 8000)

    def close(self):
        self.lp.reset()
        self.lp.close()

    def update_frame(self):
        self.next_frame = render(self.mirror_state, self.modifiers, self.player_id * 8)
        r = np.where(self.next_frame != self.frame)
        xs, ys, _ = r
        # Ignore duplicates.
        points = set(zip(xs, ys))
        for x, y in points:
            color = self.next_frame[x, y]
            self.lp.set_led(x, y, *color)
        self.frame = self.next_frame

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
            data = encode_bytes(self.player_id, pos, self.selected_effector)
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
            old = self.client_timestamp
            self.client_timestamp += timestamp - self.server_timestamp
            self.server_timestamp = timestamp
            # print('server', timestamp, self.server_timestamp, timestamp - self.server_timestamp)
            delay = self.client_timestamp - now
            # print('client', old, self.client_timestamp, now, delay)
            if delay < 0:
                print(f'late by {-delay}. jitter exceeded {self.packet_jitter_buffer} seconds')
                delay = 0

            self.mirror_state.grid = np.zeros(
                self.mirror_state.grid.shape, dtype=np.int)
            self.mirror_state.cursors = [
                cursors.Cursor(*d) for d in data['cursors']]
            for x, y, value in data.get('grid', []):
                self.mirror_state.grid[x, y] = value
            for event in data.get('events', []):
                print(f'Event: {event}')
                event[0] = event[0].encode('utf8')
                # Fractional event timing offset + jitter-prevention delay = Patch triggering delay
                # TODO: consider using this for lp display as well, to smooth frac_pos/dt-related jitter.
                event[-1] += delay
                self.client.send_message(b'/cursors', event)
                print(event)

    def run(self):
        while True:
            self.next_frame = self.frame.copy()
            self.poll_server()
            event = self.lp.get_button_event()
            if event:
                self.handle_input(event)

            self.update_frame()


if __name__ == '__main__':
    # Eventually, we might want to assign player numbers automatically.
    if len(sys.argv) < 3:
        exit('usage: client.py <server address> <player number>')
    c = CursorClient(int(sys.argv[2]))
    c.open(sys.argv[1])
    try:
        c.run()
    except KeyboardInterrupt:
        pass
    finally:
        c.close()
