"""The brain-to-board link: UDP on localhost, one JSON object per datagram.

Brain -> board, every brain tick (20 Hz):
    {"seq": n, "vx": m/s, "vy": m/s, "w": rad/s, "mode": "walk"}
    mode is what the brain asks for: "stand", "walk", "lie" or "limp".
Board -> brain, 50 Hz:
    {"t": s, "x": m, "y": m, "z": m, "yaw": rad, "mode": "STAND",
     "tripped": null or why, "walker": true/false,
     "lidar": null (none fitted) or {"front": m, "back": m, "left": m, "right": m,
              "age": s}, free distance from the body, null where nothing is seen;
              go2_board.lidar.CloudLink gets the points themselves, on 9872

UDP because neither side may ever wait on the other: a lost datagram is
replaced by the next one 50 ms later, and the board treats silence as an
order to stop (CMD_TIMEOUT_S), not as something to wait out. ROS 2 topics
(/cmd_vel, /odom) can wrap this later without changing either end.
"""
import json
import socket
import time

BOARD_PORT = 9870         # the board listens here for commands
BRAIN_PORT = 9871         # the brain listens here for state
HOST = "127.0.0.1"
CMD_TIMEOUT_S = 0.25      # 5 brain ticks without a command: velocity goes to zero
MODES = ("stand", "walk", "lie", "limp")


class Command:
    def __init__(self, vx=0.0, vy=0.0, w=0.0, mode=None, seq=-1, stamp=0.0):
        self.vx, self.vy, self.w = vx, vy, w
        self.mode, self.seq, self.stamp = mode, seq, stamp


def _socket(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, port))
    s.setblocking(False)
    return s


class BoardLink:
    """The board's end: polled once a tick, never blocks."""

    def __init__(self, port=BOARD_PORT, brain_port=BRAIN_PORT, clock=time.monotonic):
        self.sock = _socket(port)
        self.brain = (HOST, brain_port)
        self.clock = clock
        self.last = Command()

    def poll(self):
        """Drain the socket; the newest valid command wins."""
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError):
                break
            try:
                m = json.loads(data)
                mode = m.get("mode")
                cmd = Command(float(m["vx"]), float(m["vy"]), float(m["w"]),
                              mode if mode in MODES else None, int(m["seq"]), self.clock())
            except (ValueError, KeyError, TypeError):
                continue                      # a bad datagram is dropped, not obeyed
            if cmd.seq > self.last.seq or cmd.seq < self.last.seq - 1000:   # restarted brain
                self.last = cmd
        return self.last

    def velocity(self):
        """(vx, vy, w) to walk at now: zero once the brain has gone quiet."""
        c = self.last
        if self.clock() - c.stamp > CMD_TIMEOUT_S:
            return 0.0, 0.0, 0.0
        return c.vx, c.vy, c.w

    def send_state(self, **state):
        try:
            self.sock.sendto(json.dumps(state).encode(), self.brain)
        except OSError:
            pass                              # nobody listening yet

    def close(self):
        self.sock.close()


class BrainLink:
    """The brain's end (Cyberdog's Go2 robot class uses this)."""

    def __init__(self, port=BRAIN_PORT, board_port=BOARD_PORT):
        self.sock = _socket(port)
        self.board = (HOST, board_port)
        self.seq = 0
        self.state = None

    def send(self, vx, vy, w, mode="walk"):
        self.seq += 1
        msg = {"seq": self.seq, "vx": vx, "vy": vy, "w": w, "mode": mode}
        self.sock.sendto(json.dumps(msg).encode(), self.board)

    def poll(self):
        """The newest state the board has sent, or None before the first."""
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError):
                return self.state
            try:
                self.state = json.loads(data)
            except ValueError:
                continue

    def close(self):
        self.sock.close()
