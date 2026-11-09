"""The LowCmd CRC without unitree_sdk2py's pure-Python loop.

On Linux the SDK computes the CRC in C (lib/crc_aarch64.so on the J4012). On
macOS it falls back to a bit-by-bit Python loop: 1.5 ms of the 2 ms tick, and
under load the board loop then runs late enough to trip on stale state or
let the walking policy fall.

The SDK's CRC is CRC-32/MPEG-2 (poly 0x04C11DB7, init 0xFFFFFFFF, not
reflected, no final xor) over the message's 32-bit words, each fed most
significant bit first. zlib's crc32 is the reflected relative of the same
polynomial, so: reverse the bits of every byte, run zlib (C), undo its final
xor and reverse the result. The packing into words stays the SDK's own, so
the bytes covered are exactly the ones the motor side checks.
"""
import struct
import zlib

_REV = bytes(int(f"{b:08b}"[::-1], 2) for b in range(256))


def _rev32(x):
    return int(f"{x:032b}"[::-1], 2)


def crc32_words(words):
    """The SDK's crc32 over a list of uint32 words."""
    data = struct.pack(f">{len(words)}I", *words).translate(_REV)
    return _rev32(~zlib.crc32(data) & 0xFFFFFFFF)


def lowcmd_crc(sdk_crc, msg):
    """What sdk_crc.Crc(msg) returns for a unitree_go LowCmd_, fast."""
    return crc32_words(sdk_crc._CRC__PackLowCmd(msg))
