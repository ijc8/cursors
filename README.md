# cursors

instrument for FaMLE

## Setup

### Get this repo

If you're getting this project on its own:

    git clone https://github.com/ijc8/cursors && cd cursors

If you're getting it as a submodule in the [FaMLE repo](https://github.com/collaborative-music-lab/MLE/):

    git clone https://github.com/collaborative-music-lab/MLE/  # Skip this if you have it already
    cd MLE
    git submodule init && git submodule update
    cd compositions/S20/Telematic/cursors

### Configure your environment

    # Setup your environment:
    pip install virtualenv  # only if you don't already have it
    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt

### Run

Make sure your launchpad is plugged in and run:

    python client.py <server address>

If you want to run the server:

    python server.py 2  # 2 = number of players (= number of adjacent 8x8 grids)

## Misc

Currently, OSC messages are sent to `localhost:8000` on `/cursors`.
The contents of the messages are effector type (string), x, y, cursor size, cursor speed (all ints), delay time (float)
