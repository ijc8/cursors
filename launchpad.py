#!/usr/bin/env python
# Used https://github.com/dhilowitz/launchpad_rtmidi.py for reference; heavily modified.
import rtmidi
import time


# Find MIDI devices containing a given name.
# Returns a tuple of ([inputs], [outputs]), with devices specified by index.
def find_midi_devices(name):
    # Collect matching outputs.
    midi_in = rtmidi.MidiIn()
    midi_out = rtmidi.MidiOut()
    inputs = [i for i, port in enumerate(
        midi_in.get_ports()) if port.lower().find(name.lower()) >= 0]
    outputs = [i for i, port in enumerate(
        midi_out.get_ports()) if port.lower().find(name.lower()) >= 0]
    return (inputs, outputs)


def encode_color(red, green):
    assert(0 <= red <= 3 and 0 <= green <= 3)
    return red | (green << 4)

def encode_color3(red, green):
    # Convert launchpad 2 colors to launchpad 3 palette.
    assert(0 <= red <= 3 and 0 <= green <= 3)
    # TODO improve mapping
    colors = [
        [0, 19, 18, 17],
        [7, 83, 62, 13],
        [6, 11, 61, 9],
        [5, 72, 84, 96],
    ]
    return colors[red][green]

class LaunchpadException(Exception):
    pass


def encode_pos(x, y):
    assert(0 <= x <= 8)
    assert(-1 <= y <= 7)
    if y < 0:
        return 200 + x
    return x + y * 16


def decode_pos(p):
    # Call the top row buttons the row -1, since they are above the rest of the grid.
    if p >= 200:
        return (p - 200, 0)
    # Implicitly call the column buttons column 8, since they are to the right of the rest of the grid.
    return (p % 16, p // 16)


def pos2_to_3(p):
    r = p // 16
    c = p % 16
    r = 7 - r
    return (r + 1) * 10 + c + 1


def encode_pos3(x, y):
    assert(0 <= x <= 8)
    assert(-1 <= y <= 7)
    return (8 - y) * 10 + x + 1


def decode_pos3(p):
    return (p % 10 - 1, 8 - p // 10)


def test_pos():
    pairs = [((0, 0), 0),
             ((8, 0), 8),
             ((0, 7), 112),
             ((8, 7), 120),
             ((0, -1), 200)]
    for pos, p in pairs:
        assert encode_pos(*pos) == p
        assert decode_pos(p) == pos


def test_pos3():
    pairs = [((0, 0), 81),
             ((8, 0), 89),
             ((0, 7), 11),
             ((8, 7), 19),
             ((0, -1), 91)]
    for pos, p in pairs:
        assert encode_pos3(*pos) == p
        assert decode_pos3(p) == pos


# For 2-color Launchpads with 8x8 matrix and 2x8 top/right rows
class Launchpad:
    def __init__(self):
        self.midi_in = None
        self.midi_out = None

    def __del__(self):
        self.close()

    def open(self, name="Launchpad"):
        ins, outs = find_midi_devices(name)
        if not ins or not outs:
            raise LaunchpadException("No launchpad detected.")
        if "MK3" in rtmidi.MidiOut().get_port_name(outs[0]):
            # Hack to emulate mk2 on mk3.
            ins, outs = find_midi_devices("MK3 MIDI")
            self.midi_in = rtmidi.MidiIn().open_port(ins[0])
            self.midi_out = rtmidi.MidiOut().open_port(outs[0])
            print("Mk3 detected, emulating Mk2.")
            # Programmer mode
            self.midi_out.send_message([240, 0, 32, 41, 2, 13, 14, 1, 247])
            self.encode_color = encode_color3
            self.encode_pos = encode_pos3
            self.decode_pos = decode_pos3
        else:
            self.midi_in = rtmidi.MidiIn().open_port(ins[0])
            self.midi_out = rtmidi.MidiOut().open_port(outs[0])
            self.encode_color = encode_color
            self.encode_pos = encode_pos
            self.decode_pos = decode_pos

    def close(self):
        if self.midi_in:
            self.midi_in.close_port()
            self.midi_in = None
        if self.midi_out:
            self.midi_out.close_port()
            self.midi_out = None

    # TODO: check if intermediate values (between 0 and 127) do anything useful.
    # perhaps consolidate into one method.
    # Reset the Launchpad, turn of all LEDs.
    def reset(self):
        self.midi_out.send_message([176, 0, 0])

    # Turn all LEDs on.
    def enable_all_leds(self):
        self.midi_out.send_message([176, 0, 127])

    # Controls a grid LED by its x, y position; with <green/red> brightness: 0..3
    def set_led(self, x, y, red, green):
        pos = self.encode_pos(x, y)
        color = self.encode_color(red, green)
        self._set_led_raw(pos, color)

    def _set_led_raw(self, pos, color):
        if pos > 199:
            assert(200 <= pos <= 207)
            # Different command for top row of buttons.
            self.midi_out.send_message([176, pos - 96, color])
        else:
            assert(0 <= pos <= 120)
            self.midi_out.send_message([144, pos, color])

    # Sends a list of consecutive color values to the Launchpad.
    # Requires less than half of the commands to update consecutive buttons,
    # vs. repeated calls to set_led().
    # First, the 8x8 matrix is updated, left to right, top to bottom,
    # followed by the rightmost buttons and the top "automap" buttons.
    # Note that the amount of LEDs needs to be even.
    # Colors should be a list of pairs (red, green), as passed to set_led().
    # TODO: inherited this; check if it actually works.
    def set_led_batch(self, colors):
        assert(len(colors) % 2 == 0)
        for i in range(0, len(colors), 2):
            self.midi_out.send_message(
                [146, encode_color(*colors[i]), encode_color(*colors[i+1])])

    # "Homes" the next set_led_batch() call, so it will start with the first LED again.
    # TODO: can we set the position arbitrarily?
    def reset_led_batch_pos(self):
        self.midi_out.send_message([176, 1, 0])

    # Returns the last button press/release as a tuple (pos, state).
    # If there have been no changes since the last call, returns None instead.
    def get_button_event(self):
        m = self.midi_in.get_message()
        if m:
            pos = m[0][1] if m[0][0] in (144, 176) else m[0][1] + 96
            state = True if m[0][2] > 0 else False
            return (self.decode_pos(pos), state)
        return None

    # Clears the button buffer
    def flush_button_events(self):
        while self.get_button_event():
            pass
