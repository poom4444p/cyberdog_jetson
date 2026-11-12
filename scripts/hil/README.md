# Hardware in the loop: the J4012 as the dog

Everything the dog would carry runs on the J4012; the world it walks in runs
on the Mac. They talk over the USB-C link, in the messages a real Go2 speaks.

| Where | What | How it starts |
|---|---|---|
| J4012, cores 6-7 | go2_board `board_main`: motors (DDS), safety, the walking policy, the Mid-360 guard | `onboard.sh board` |
| J4012, cores 0-5 | the Cyberdog brain (`run_building --robot go2_board`) | `onboard.sh brain DEST` |
| J4012, GPU | VAMOS | `onboard.sh vamos` |
| Mac | go2_world: physics, people, the lift (`run_world.py`); the Mid-360 (`run_lidar.py`) | `run_hil.sh` |

## Files

| File | Runs on | |
|---|---|---|
| `run_hil.sh` | Mac | a whole run: world, LiDAR, board, brain; logs and a summary |
| `onboard.sh` | J4012 | starts and stops the board's side; `run_hil.sh` calls it over ssh |
| `trace_run_building.py` | J4012 | the brain, printing its commands (`TRACE=1`) |
| `dds_gaps.py` | either | how regularly states and commands arrive on the link |

## Setting up a fresh J4012

Once, after flashing JetPack 6.2. The board needs the internet for the apt and
pip installs of the environments; on the rig it has none (only the USB-C link
to the Mac), so this repo's own packages are installed offline, last.

```
# Mac: ssh without a password (the scripts use BatchMode)
ssh-copy-id jittarin@192.168.55.1

# board: packages, 25W mode (its ID: grep "POWER_MODEL ID" /etc/nvpmodel.conf)
sudo apt install -y git cmake build-essential python3-venv python3-dev curl
sudo nvpmodel -m <25W ID>

# board: CycloneDDS 0.10.x, then unitree_sdk2py in ~/go2env
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x ~/cyclonedds
cd ~/cyclonedds && mkdir -p build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_TESTING=OFF && cmake --build . --target install -j6
python3 -m venv ~/go2env && . ~/go2env/bin/activate
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git ~/unitree_sdk2_python
CYCLONEDDS_HOME=~/cyclonedds/install pip install -e ~/unitree_sdk2_python
pip install scipy PyYAML Pillow matplotlib requests mujoco "imageio[ffmpeg]" pytest

# board: VAMOS in ~/vamosenv, torch from NVIDIA's JetPack 6 index
python3 -m venv ~/vamosenv && . ~/vamosenv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install "numpy<2" transformers==5.17.0 peft==0.20.0 accelerate==1.15.0 fastapi uvicorn python-multipart pillow huggingface_hub

# Mac: a setuptools new enough for editable installs (see below), from the Mac's internet
pip download --no-deps --only-binary=:all: --python-version 3.10 setuptools==75.8.0 -d /tmp/wheels
scp /tmp/wheels/setuptools-75.8.0-py3-none-any.whl jittarin@192.168.55.1:/tmp/
ssh jittarin@192.168.55.1 '. ~/go2env/bin/activate; pip install --no-index /tmp/setuptools-75.8.0-py3-none-any.whl'

# Mac: this repo (go2_board/ and its go2_flat_v5 policy are in it), installed with no internet
bash scripts/hil/run_hil.sh sync
ssh jittarin@192.168.55.1 '. ~/go2env/bin/activate; pip install --no-deps --no-build-isolation -e ~/cyberdog_jetson/go2_board -e ~/cyberdog_jetson'

# Mac: the Go2 model, then the building scene built on the board (output/ is not synced)
ssh jittarin@192.168.55.1 'mkdir -p ~/mujoco_menagerie'
rsync -a "${MENAGERIE_PATH:-$HOME/mujoco_menagerie}/unitree_go2" jittarin@192.168.55.1:~/mujoco_menagerie/
ssh jittarin@192.168.55.1 '. ~/go2env/bin/activate; cd ~/cyberdog_jetson && python -m cyberdog.sim.scene.build_scene --building'

# Mac: VAMOS's server and weights
ssh jittarin@192.168.55.1 'mkdir -p ~/vamos ~/.cache/huggingface/hub'
rsync -a --exclude __pycache__ vendor/VAMOS/server jittarin@192.168.55.1:~/vamos/
rsync -a ~/.cache/huggingface/hub/models--mateoguaman--vamos jittarin@192.168.55.1:~/.cache/huggingface/hub/
```

Why the setuptools step and the two flags: `pip install -e` normally builds in
an isolated environment and downloads `setuptools>=64` into it, which fails
with no internet ("Failed to build ... when installing build dependencies").
`--no-build-isolation` uses the venv's own setuptools instead, but Ubuntu
22.04's venv comes with 59.6, which has no PEP 660 `build_editable` hook
("missing the 'build_editable' hook"); 75.8.0 has it. `--no-deps` keeps pip
from looking for the packages already installed above. After a sync that
changes `pyproject.toml`, run the install line again.

`output/scene_cache/building.xml` is what the brain walks in; it names the
board's own `~/mujoco_menagerie`, so it is built there, not copied from the
Mac. Rebuild it after moving the menagerie or changing the building.

The Mid-360's receive buffer: a real unit (or `LIDAR_ARGS=--dense`) sends
~300 KB a frame, ten frames a second, and Linux caps a socket's receive
buffer at `net.core.rmem_max`, 212,992 bytes out of the box, without a word.
Left there, the board lost 12 % of the packets in a dense route and half its
frames arrived partial (HIL 11-11 15:39). Once, with sudo:

```bash
echo net.core.rmem_max=16777216 | sudo tee /etc/sysctl.d/90-livox.conf
sudo sysctl --system | grep rmem_max      # net.core.rmem_max = 16777216
```

The file keeps it across boots. `board_main` asks for 8 MB
(`go2_board.livox.RCVBUF`), prints what it got (`[board] Mid-360 receive
buffer 8.0 MB`) and warns, with this fix, when the kernel gave less.

Then `run_hil.sh clock`, `walk` and `route`, as below. To switch the board off:
`ssh -t jittarin@192.168.55.1 'sudo shutdown -h now'`, and wait for it to go dark.

## A run

```
bash scripts/hil/run_hil.sh sync                 # after changing the brain or go2_board/
bash scripts/hil/run_hil.sh clock                # after the board boots (asks for its sudo password)
PY=/opt/anaconda3/envs/unitree_sim/bin/python bash scripts/hil/run_hil.sh walk
PY=/opt/anaconda3/envs/unitree_sim/bin/python bash scripts/hil/run_hil.sh route "server room"
```

`PY` is a Python with go2_world's packages (mujoco, unitree_sdk2py). Logs go to
`output/hil/<time>-<walk|route>/`.

`sync` copies this repo **as checked out** to `~/cyberdog_jetson` on the board:
it prints the branch and commit and asks before copying (`SYNC_YES=1` when
there is no terminal). The brain and go2_board used to be two repos synced
separately, and a sync from a checkout without the board fixes once put an old
board on the J4012, which only a fall showed; one repo means one commit for both.

- **`walk`** (about a minute): the board stands, walks 4 s at 0.54 m/s and
  0.47 rad/s, and stands again, driven by go2_board's `brain_cli.py`, no brain.
  It ends `RESULT: walked` or `RESULT: FELL`. Run it first after any change to
  go2_board: it is what showed the board steering on states 0.3 s old.
- **`route DEST`** (3-4 minutes): the whole route with the brain and VAMOS, e.g.
  floor 1 to the server room on floor 3, with the lift and four people. The
  board steers on its own odometry (`POSE=lio`, the default since 11-12:
  small_gicp on the Mid-360's sweeps with the gyro's rotation, as the real dog
  has to), not the world's true pose; `POSE=sim` for that. The route checks
  small_gicp is on the board first. With lio the emulated Mid-360 runs dense
  (`DENSE=1`, ~20k points a sweep, a real unit's rate; `DENSE=0` for the old
  ~2k pattern): on the sparse one the odometry drifted 1.02 m against 0.26 m. Three lio routes before the switch (11-11,
  11-12, 0 and 4 people) all arrived, at most 0.37 m off the truth.

Both end with what `dds_gaps.py` saw on the board's side of the link:

```
lowstate (world)    500/s  median  2.0  p99   2.6  ...
lowcmd (board)      466/s  median  2.0  p99   3.5  ...
```

`lowcmd` is the board loop's rate (target 500/s). A separate process with its
own socket, it shows what is on the wire, not how fresh the states the board
uses are: those need go2_board's own check (`tests/test_sim_bus.py`).

A route then stops the board (its log's last lines: how it stopped, and with
`POSE=lio`, the default, the odometry's error against the world's pose), then the world,
and prints the world's own score of the run, on the physics itself
(go2_world's `score.py`):

```
world truth: 81s of simulated time scored
world truth: collisions: none -- no part of the dog touched a wall or an obstacle
world truth: contact: none -- closest it came to anybody was 0.84 m
```

with a `world: t=... COLLISION starts/ends` or `CONTACT starts/ends` line
for each one, in go2_world's frame (the map's is that + (21.0, 9.5)). These
are the lines to trust. The brain's own `collisions:`/`handler:` lines are
marked `[on the board's pose]`: its twin is put where the board says the dog
is, which with `POSE=lio` is off by the odometry's error. It has no people of
its own in HIL, so it no longer prints a contact distance (it printed
"closest ... 99.00 m", its starting value, in every HIL run before 11-12).
The person on the handle exists only in the brain: `handler:` stays scored
on the board's pose.

## Settings worth knowing

- **`STALE_MS`** (`onboard.sh`, default 200): the board goes limp after this long
  without a state. The Mac's world now and then pauses for 100 ms and more, so
  the rig uses 200. go2_board's own limit is 50 ms (`safety.STALE_S`); set
  `STALE_MS=` (empty) to use it. Anything driving real motors must.
- **`WORLD_ARGS=--held-pd`**: go2_world's motors as unitree_mujoco has them (the
  torque worked out once per command and held) instead of every physics step.
- **Cores**: `BOARD_CORES` (6,7) and `BRAIN_CORES` (0-5) in `onboard.sh`.
- **`RECORD=1`**: go2_world records the run to `output/hil/<time>-<walk|route>/world.mp4`
  (a camera chasing the dog, world time and position on each frame). Needs
  go2_world's `record` extra in `PY`'s env (`pip install -e ~/Work/unitree/go2_world[record]`).
  It costs the physics nothing measurable (route 11-09: lowstate max 8.5 ms).
- **`VIEWER=1`**: run the world in MuJoCo's viewer (`mjpython` next to `PY`)
  instead of headless. Watch the stale trips: the viewer makes the world pause more.

## The board's clock

The J4012 has no working RTC (`timedatectl`: RTC time 1970). After a boot its
clock is wherever it was at the last shutdown, until NTP sets it -- and with no
network but the USB-C link, nothing does. The rig runs on monotonic clocks, so
a wrong clock spoils only log times; `run_hil.sh` warns when it is more than
2 s off, and `run_hil.sh clock` sets it from the Mac (sudo on the board, so it
asks for the password). To set it without a password, a sudoers rule on the
board can allow `date` alone, e.g. with `sudo visudo -f /etc/sudoers.d/date`:

```
jittarin ALL=(root) NOPASSWD: /usr/bin/date
```

That is a decision for whoever runs the board; the scripts do not make it.
