"""livox.py against packets built here byte by byte from Livox's table, not
from go2_world's encoder: no sensor, no emulator."""
import socket
import struct
import time
import zlib

import numpy as np
import pytest

from go2_board import livox
from go2_board.livox import FrameAssembler, LivoxError, LivoxReceiver, parse

MS = 1_000_000


def raw(data_type, payload, dot_num, udp_cnt=0, frame_cnt=0, stamp=0, interval=0):
    """A packet laid out by offset, as the protocol doc's table has it."""
    body = struct.pack("<Q", stamp) + payload
    head = bytearray(28)
    head[0] = 0                                               # version
    struct.pack_into("<H", head, 1, 36 + len(payload))        # length
    struct.pack_into("<H", head, 3, interval)                 # time_interval, 0.1 us
    struct.pack_into("<H", head, 5, dot_num)
    struct.pack_into("<H", head, 7, udp_cnt)
    head[9], head[10], head[11] = frame_cnt, data_type, 0
    struct.pack_into("<I", head, 24, zlib.crc32(body))
    return bytes(head) + body


def pts(xyz_mm, refl=100):
    return b"".join(struct.pack("<iiiBB", *p, refl, 0) for p in xyz_mm)


def cloud(n=96, x=1000):
    return pts([(x, i, 0) for i in range(n)])


def pkt(udp_cnt, stamp, frame_cnt=0, n=96):
    return parse(raw(1, cloud(n), n, udp_cnt, frame_cnt, stamp, interval=10_000))   # 1 ms a packet


# -- one packet -------------------------------------------------------------

def test_points_in_metres_and_timed_like_the_driver():
    p = parse(raw(1, pts([(1500, -250, 40), (0, 0, 0)], refl=77), 2,
                  udp_cnt=3, frame_cnt=9, stamp=10 * MS, interval=20))
    assert (p.udp_cnt, p.frame_cnt, p.dot_num) == (3, 9, 2)
    xyz, refl, tag, t = livox.points(p)
    assert np.allclose(xyz[0], [1.5, -0.25, 0.04]) and np.all(xyz[1] == 0)
    assert refl[0] == 77
    assert list(t) == [10 * MS, 10 * MS + 20 * 100 // 2]     # stamp + i * interval*100/dot_num


def test_low_precision_is_centimetres():
    p = parse(raw(2, struct.pack("<hhhBB", 150, -25, 4, 10, 0), 1))
    assert np.allclose(livox.points(p)[0][0], [1.5, -0.25, 0.04])


def test_imu_g_to_metres_per_second_squared():
    p = parse(raw(0, struct.pack("<6f", 0.1, 0.2, 0.3, 0.0, 0.0, 1.0), 1, stamp=5))
    s = livox.imu(p)
    assert s.t == 5 and np.allclose(s.gyro, [0.1, 0.2, 0.3]) and np.isclose(s.acc[2], livox.G)


@pytest.mark.parametrize("hurt, why", [
    (lambda b: b[:-1], "length"),
    (lambda b: b[:30] + bytes([b[30] ^ 1]) + b[31:], "crc32"),      # a timestamp bit
    (lambda b: b[:-1] + bytes([b[-1] ^ 1]), "crc32"),               # a data bit
    (lambda b: b"\x01" + b[1:], "version"),
    (lambda b: b[:10] + b"\x07" + b[11:], "data_type"),
    (lambda b: b[:20], "short"),
])
def test_bad_packets_are_refused(hurt, why):
    with pytest.raises(LivoxError, match=why):
        parse(hurt(raw(1, cloud(2), 2)))


def test_dot_num_must_match_the_bytes():
    with pytest.raises(LivoxError, match="points"):
        parse(raw(1, cloud(3), 2))


# -- frames -----------------------------------------------------------------

def feed_all(a, packets):
    return [f for f in (a.feed(p) for p in packets) if f is not None]


def frame(start_ms, n=5, frame_cnt=0, skip=()):
    return [pkt(i, (start_ms + i) * MS, frame_cnt) for i in range(n) if i not in skip]


def test_frames_cut_where_udp_cnt_starts_again():
    a = FrameAssembler()
    out = feed_all(a, frame(0, frame_cnt=0) + frame(100, frame_cnt=1) + frame(200, frame_cnt=2))
    assert len(out) == 2 and a.flush().frame_cnt == 2
    assert all(f.packets == 5 and f.missing == 0 for f in out)
    assert len(out[0].xyz) == 5 * 96 and out[0].stamp == 0
    assert a.lost == 0 and a.frames_skipped == 0


def test_gaps_counted_in_the_middle_at_the_start_and_off_the_end():
    a = FrameAssembler()
    out = feed_all(a, frame(0) + frame(100, skip={2}) + frame(200, skip={0}) + frame(300, skip={4}))
    out.append(a.flush())
    assert [f.missing for f in out] == [0, 1, 1, 1]
    assert a.lost == 3


def test_late_packet_rejoins_its_frame():
    a = FrameAssembler()
    f = frame(0)
    out = feed_all(a, [f[0], f[1], f[3], f[2], f[4]] + frame(100))
    assert out[0].packets == 5 and out[0].missing == 0 and a.late == 0


def test_late_first_packet_still_belongs_to_its_frame():
    a = FrameAssembler()
    f = frame(100)
    out = feed_all(a, frame(0) + [f[1], f[0]] + f[2:] + frame(200))
    assert out[1].packets == 5 and out[1].missing == 0 and a.late == 0


def test_packet_too_late_for_its_frame_is_dropped():
    a = FrameAssembler()
    f1, f2 = frame(100), frame(200)
    out = feed_all(a, frame(0) + f1[:4] + f2[:2] + [f1[4]] + f2[2:] + frame(300))
    assert a.late == 1 and out[1].missing == 1 and out[2].missing == 0


def test_lost_boundary_still_cuts_on_time():
    a = FrameAssembler()                          # frame 0 loses its tail, frame 1 its head
    out = feed_all(a, frame(0, n=10)[:3] + frame(100, n=10)[5:])
    assert len(out) == 1 and out[0].packets == 3


def test_frame_cnt_jumps_count_only_when_it_moves():
    a = FrameAssembler()
    feed_all(a, frame(0, frame_cnt=1) + frame(100, frame_cnt=4) + frame(200, frame_cnt=4))
    a.flush()
    assert a.frames_skipped == 2


def test_no_return_points_dropped_unless_kept():
    p = parse(raw(1, pts([(1000, 0, 0), (0, 0, 0)]), 2))
    a, k = FrameAssembler(), FrameAssembler(keep_empty=True)
    a.feed(p), k.feed(p)
    assert len(a.flush().xyz) == 1 and len(k.flush().xyz) == 2


# -- over UDP ---------------------------------------------------------------

def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_receiver_over_loopback():
    pp, ip = free_port(), free_port()
    rx = LivoxReceiver("127.0.0.1", pp, ip)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for i in range(3):
            tx.sendto(raw(1, cloud(), 96, i, 0, i * MS), ("127.0.0.1", pp))
        tx.sendto(b"\x00" * 40, ("127.0.0.1", pp))                              # garbage
        tx.sendto(raw(1, cloud(), 96, 0, 1, 100 * MS), ("127.0.0.1", pp))       # next frame
        tx.sendto(raw(0, struct.pack("<6f", 0, 0, 0, 0, 0, 1), 1), ("127.0.0.1", ip))
        frames, imu = [], []
        end = time.monotonic() + 1.0
        while time.monotonic() < end and not (frames and imu):
            f, s = rx.poll()
            frames += f
            imu += s
            time.sleep(0.005)
        assert len(frames) == 1 and frames[0].packets == 3 and len(imu) == 1
        assert rx.stats()["bad"] == 1
    finally:
        tx.close()
        rx.close()


def test_a_sensor_that_starts_again_is_followed_not_dropped():
    # time since power-on: a restart sends the clock back to zero mid-frame
    a = FrameAssembler()
    old = feed_all(a, frame(60_000) + frame(60_100)[:3])          # a frame open when it stopped
    new = feed_all(a, frame(0) + frame(100) + frame(200))
    assert len(old) == 1 and a.restarts == 1
    assert len(new) == 2 and new[0].stamp == 0 and new[0].packets == 5 and a.late == 0


# --- the receive buffer (RCVBUF) ----------------------------------------------

class FakeSock:
    """setsockopt refuses anything over `limit` (macOS), or caps silently at
    `cap` and reports twice that (Linux)."""
    def __init__(self, limit=None, cap=None, doubles=False):
        self.limit, self.cap, self.doubles, self.size = limit, cap, doubles, 0

    def setsockopt(self, level, opt, value):
        if self.limit is not None and value > self.limit:
            raise OSError(55, "No buffer space available")
        self.size = value if self.cap is None else min(value, self.cap)

    def getsockopt(self, level, opt):
        return 2 * self.size if self.doubles else self.size


def test_a_refused_size_is_halved_until_one_is_taken(monkeypatch):
    from go2_board import livox
    monkeypatch.setattr(livox.sys, "platform", "darwin")
    assert livox._ask_rcvbuf(FakeSock(limit=3 << 20), 8 << 20) == 2 << 20


def test_linux_silently_capped_is_seen_as_capped(monkeypatch):
    # The J4012 before net.core.rmem_max was raised: 212,992 bytes, reported doubled.
    from go2_board import livox
    monkeypatch.setattr(livox.sys, "platform", "linux")
    got = livox._ask_rcvbuf(FakeSock(cap=212_992, doubles=True), livox.RCVBUF)
    assert got == 212_992 and got < livox.RCVBUF // 2
    assert livox._ask_rcvbuf(FakeSock(cap=16 << 20, doubles=True), livox.RCVBUF) == livox.RCVBUF


def test_the_receiver_reports_what_it_was_granted():
    from go2_board.livox import RCVBUF
    rx = LivoxReceiver("127.0.0.1", 19761, 19762, rcvbuf=1 << 20)
    try:
        assert 0 < rx.rcvbuf <= 1 << 20 and rx.stats()["rcvbuf"] == rx.rcvbuf
        assert rx.rcvbuf_short == (rx.rcvbuf < (1 << 20) // 2)
        assert RCVBUF >= 4 << 20                       # ~300 KB a dense frame, with room
    finally:
        rx.close()
