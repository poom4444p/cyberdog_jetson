"""Mid-360 packets in, points out: Livox SDK2's UDP format, read directly.

The same code reads go2_world's emulator on the Mac and the real Mid-360 on
the J4012: both send points 56300 -> 56301 and IMU 56400 -> 56401. Format
from Livox's protocol doc for the Mid-360 and Livox-SDK2's livox_lidar_def.h:

  off size field
   0   1   version         0
   1   2   length          the whole UDP payload
   3   2   time_interval   first point to last, 0.1 us
   5   2   dot_num         points in the packet
   7   2   udp_cnt         packet count in the frame, 0 at frame start
   9   1   frame_cnt       +1 a frame; Livox: not valid for non-repetitive scan
  10   1   data_type       0 IMU, 1 xyz int32 mm, 2 xyz int16 cm
  11   1   time_type       0 since power-on, 1 PTP/gPTP, 2 GPS
  12  12   reserved
  24   4   crc32           over timestamp + data (zlib's CRC-32)
  28   8   timestamp       ns, first point
  36       data            14 B a point (type 1), 8 B (type 2), 24 B IMU

Frames are cut where udp_cnt starts again, not on frame_cnt, which the Mid-360
does not keep (livox_ros_driver2 does not use it either). Gaps are counted
from udp_cnt. Point i of a packet is at timestamp + i * time_interval * 100 /
dot_num ns, which is what livox_ros_driver2 does.

A point with no return is (0, 0, 0) on the wire; frames drop those unless
asked to keep them. IMU acceleration is in g on the wire, gyro in rad/s.
"""
import socket
import struct
import zlib

import numpy as np

POINT_PORT = 56301           # host side
IMU_PORT = 56401
HEADER = struct.Struct("<BHHHHBBB12sIQ")
HEADER_SIZE = 36
IMU, CARTESIAN_HIGH, CARTESIAN_LOW = 0, 1, 2
POINT = {
    CARTESIAN_HIGH: (np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                               ("reflectivity", "u1"), ("tag", "u1")]), 1e-3),
    CARTESIAN_LOW: (np.dtype([("x", "<i2"), ("y", "<i2"), ("z", "<i2"),
                              ("reflectivity", "u1"), ("tag", "u1")]), 1e-2),
}
IMU_SAMPLE = struct.Struct("<6f")
G = 9.81


class LivoxError(ValueError):
    """A packet that is not one: wrong length, version, CRC or type."""


class Packet:
    __slots__ = ("time_interval", "dot_num", "udp_cnt", "frame_cnt", "data_type",
                 "time_type", "timestamp", "data")

    def __init__(self, time_interval, dot_num, udp_cnt, frame_cnt, data_type, time_type,
                 timestamp, data):
        self.time_interval, self.dot_num = time_interval, dot_num
        self.udp_cnt, self.frame_cnt = udp_cnt, frame_cnt
        self.data_type, self.time_type = data_type, time_type
        self.timestamp, self.data = timestamp, data


def parse(buf):
    """bytes -> Packet, or LivoxError. Checks everything the header lets us."""
    if len(buf) < HEADER_SIZE:
        raise LivoxError(f"short: {len(buf)} bytes")
    (version, length, interval, dot_num, udp_cnt, frame_cnt, data_type, time_type,
     _, crc, stamp) = HEADER.unpack_from(buf)
    if version != 0:
        raise LivoxError(f"version {version}")
    if length != len(buf):
        raise LivoxError(f"length field {length}, datagram {len(buf)}")
    if zlib.crc32(memoryview(buf)[28:]) != crc:
        raise LivoxError("crc32")
    if data_type == IMU:
        size = IMU_SAMPLE.size
    elif data_type in POINT:
        size = POINT[data_type][0].itemsize
    else:
        raise LivoxError(f"data_type {data_type} not handled")
    if len(buf) - HEADER_SIZE != dot_num * size:
        raise LivoxError(f"{dot_num} points of type {data_type} in {len(buf) - HEADER_SIZE} bytes")
    return Packet(interval, dot_num, udp_cnt, frame_cnt, data_type, time_type, stamp,
                  bytes(buf[HEADER_SIZE:]))


def points(pkt):
    """(xyz m (n, 3), reflectivity (n,), tag (n,), t ns (n,)) from a point packet."""
    dtype, scale = POINT[pkt.data_type]
    raw = np.frombuffer(pkt.data, dtype)
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1) * scale
    dt = pkt.time_interval * 100 // max(pkt.dot_num, 1)
    t = pkt.timestamp + dt * np.arange(pkt.dot_num, dtype=np.int64)
    return xyz, raw["reflectivity"].copy(), raw["tag"].copy(), t


class ImuSample:
    __slots__ = ("t", "gyro", "acc")

    def __init__(self, t, gyro, acc):
        self.t, self.gyro, self.acc = t, gyro, acc     # ns; rad/s; m/s^2


def imu(pkt):
    v = IMU_SAMPLE.unpack(pkt.data[:IMU_SAMPLE.size])
    return ImuSample(pkt.timestamp, np.array(v[:3]), np.array(v[3:]) * G)


class Frame:
    def __init__(self, xyz, reflectivity, tag, t, stamp, frame_cnt, packets, missing):
        self.xyz, self.reflectivity, self.tag, self.t = xyz, reflectivity, tag, t
        self.stamp = stamp            # ns, the first packet received
        self.frame_cnt = frame_cnt
        self.packets = packets        # received
        self.missing = missing        # known lost, from udp_cnt


class FrameAssembler:
    """Point packets in, whole frames out.

    A frame closes when udp_cnt starts again with a newer timestamp, or when
    a packet is a whole frame_ns (one 10 Hz frame) after the frame began (in case the end of one
    frame and the start of the next were both lost). A packet
    whose udp_cnt is behind but whose timestamp is not new is late (the network
    reordered it): it joins its frame if that is still open, and is counted
    and dropped if not. Packets lost off a frame's end are counted against the
    longest frame seen so far.

    frame_cnt is only checked for skipped frames when it moves: the Mid-360
    may leave it standing still.
    """

    # The sensor's clock is time since its power-on: further back than this is
    # a sensor that started again (power blip, or the emulator restarted), not
    # a late packet. Dropping those as late left the board blind until the new
    # clock caught up with the old one: in HIL, a minute and more.
    RESTART_NS = 1_000_000_000

    def __init__(self, keep_empty=False, frame_ns=100_000_000):
        self.keep_empty, self.frame_ns = keep_empty, frame_ns
        self.restarts = 0
        self.parts, self.next_cnt, self.start = [], 0, 0
        self.frame_cnt, self.missing, self.last_t = 0, 0, -1
        self.longest = 0
        self.lost = self.late = self.frames = self.frames_skipped = 0
        self._prev_frame_cnt = None

    def feed(self, pkt):
        """Returns the frame this packet closed, or None."""
        done = None
        if self.parts and pkt.timestamp < self.start - self.RESTART_NS:
            self.parts, self.restarts = [], self.restarts + 1   # the open frame is of the old clock
            self._prev_frame_cnt = None
        if self.parts:
            span = pkt.time_interval * 100                  # ns, one packet's worth of points
            if pkt.timestamp < self.start - span // 2:      # from a frame already closed
                self.late += 1
                return None
            if pkt.timestamp - self.start >= self.frame_ns:
                done = self._close()
            elif pkt.udp_cnt < self.next_cnt:
                if pkt.timestamp < self.last_t:             # late, and its frame is still open
                    self.missing -= 1
                    self.parts.append(pkt)
                    return None
                done = self._close()                        # udp_cnt started again: a new frame
        if not self.parts:
            # where the frame began, even if its first packets are lost or late
            self.start = pkt.timestamp - pkt.udp_cnt * pkt.time_interval * 100
            self.frame_cnt, self.next_cnt, self.missing = pkt.frame_cnt, 0, 0
        self.missing += pkt.udp_cnt - self.next_cnt     # zero unless some went missing
        self.next_cnt = pkt.udp_cnt + 1
        self.last_t = pkt.timestamp
        self.parts.append(pkt)
        return done

    def flush(self):
        """The open frame, closed now (the stream stopped)."""
        return self._close() if self.parts else None

    def _close(self):
        parts = sorted(self.parts, key=lambda p: p.udp_cnt)
        top = parts[-1].udp_cnt + 1
        tail = max(0, self.longest - top)
        self.longest = max(self.longest, top)
        missing = self.missing + tail
        self.lost += missing
        self.frames += 1
        if self._prev_frame_cnt is not None:
            step = (self.frame_cnt - self._prev_frame_cnt) % 256
            self.frames_skipped += max(0, step - 1)
        self._prev_frame_cnt = self.frame_cnt
        cols = [points(p) for p in parts]
        xyz, refl, tag, t = (np.concatenate(c) for c in zip(*cols))
        if not self.keep_empty:
            ok = np.any(xyz != 0.0, axis=1)
            xyz, refl, tag, t = xyz[ok], refl[ok], tag[ok], t[ok]
        frame = Frame(xyz, refl, tag, t, parts[0].timestamp, self.frame_cnt, len(parts), missing)
        self.parts = []
        return frame


def _socket(port, host):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)   # a dense frame is ~300 KB
    s.bind((host, port))
    s.setblocking(False)
    return s


class LivoxReceiver:
    """Both ports, polled; never blocks. host "" listens on every interface,
    which on the J4012 is what the real sensor, sending to its configured host
    IP, needs."""

    def __init__(self, host="", point_port=POINT_PORT, imu_port=IMU_PORT, keep_empty=False):
        self.points = _socket(point_port, host)
        self.imu = _socket(imu_port, host)
        self.frames = FrameAssembler(keep_empty)
        self.packets = self.bad = self.imu_count = 0
        self.last_error = None

    def _drain(self, sock):
        while True:
            try:
                data, _ = sock.recvfrom(2048)
            except (BlockingIOError, InterruptedError):
                return
            try:
                yield parse(data)
            except LivoxError as e:
                self.bad += 1
                self.last_error = str(e)

    def poll(self):
        """(frames completed since the last poll, IMU samples since the last poll)."""
        frames, samples = [], []
        for pkt in self._drain(self.points):
            self.packets += 1
            if pkt.data_type == IMU:
                continue
            f = self.frames.feed(pkt)
            if f is not None:
                frames.append(f)
        for pkt in self._drain(self.imu):
            if pkt.data_type == IMU:
                self.imu_count += 1
                samples.append(imu(pkt))
        return frames, samples

    def stats(self):
        a = self.frames
        return {"packets": self.packets, "bad": self.bad, "frames": a.frames, "lost": a.lost,
                "late": a.late, "frames_skipped": a.frames_skipped, "imu": self.imu_count,
                "last_error": self.last_error}

    def close(self):
        self.points.close()
        self.imu.close()
