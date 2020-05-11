import socket
import numpy as np
import launchpad
import select
import sys
import cursors
import json
import time

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
    def __init__(self, nickname=''):
        self.player_id = None
        self.nickname = nickname[:8]
        self.lp = launchpad.Launchpad()
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
        self.lp.open()
        self.lp.flush_button_events()
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

        self.sockf = self.socket.makefile('r')

    def close(self):
        self.lp.reset()
        self.lp.close()

    def update_frame(self):
        if self.client_timestamp is not None:
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
        while True:
            self.next_frame = self.frame.copy()
            self.poll_server()
            event = self.lp.get_button_event()
            if event:
                self.handle_input(event)

            self.update_frame()
            time.sleep(1/60)


if __name__ == '__main__':
    # Eventually, we might want to assign player numbers automatically.
    if len(sys.argv) < 2:
        exit('usage: client.py <server address> [nickname]')
    c = CursorClient(sys.argv[2] if len(sys.argv) > 2 else '')
    c.open(sys.argv[1])
    try:
        c.run()
    except KeyboardInterrupt:
        pass
    finally:
        c.close()
