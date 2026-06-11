"""
protocol.py — CRC-8 and binary packet parser.

Frame layout (STM32 → PC, ADC data):
  [0xAA][0x55]  sync
  [type  : u16 LE]
  [seq   : u32 LE]
  [count : u16 LE]
  [sample×count : u16 LE each]
  [crc8  : u8]   CRC over everything after sync

Frame layout (PC → STM32, command):
  [0xAA][0x55]
  [0x0002 : u16 LE]   PKT_TYPE_CMD
  [cmd_id : u8]
  [payload : N bytes]
  [crc8 : u8]         CRC over type + cmd_id + payload
"""
import struct
import numpy as np
from .config import PKT_TYPE_ADC, PKT_TYPE_TELEMETRY

def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc

class PacketParser:
    _S_SYNC0, _S_SYNC1, _S_HDR, _S_PAYLOAD, _S_CRC = range(5)

    def __init__(self):
        self._state   = self._S_SYNC0
        self._buf     = bytearray()
        self._hdr_buf = bytearray()
        self._type = self._seq = self._count = 0

    def feed(self, data: bytes):
        for b in (data if isinstance(data[0], int) else [ord(c) for c in data]):
            yield from self._proc(b)

    def _proc(self, b):
        s = self._state
        if s == self._S_SYNC0:
            if b == 0xAA: self._state = self._S_SYNC1
        elif s == self._S_SYNC1:
            self._state = self._S_HDR if b == 0x55 else self._S_SYNC0
            if self._state == self._S_HDR: self._hdr_buf = bytearray()
        elif s == self._S_HDR:
            self._hdr_buf.append(b)
            if len(self._hdr_buf) == 8:
                (self._type,)  = struct.unpack_from('<H', self._hdr_buf, 0)
                (self._seq,)   = struct.unpack_from('<I', self._hdr_buf, 2)
                (self._count,) = struct.unpack_from('<H', self._hdr_buf, 6)
                if 0 < self._count <= 1024:
                    self._buf = bytearray(); self._state = self._S_PAYLOAD
                else:
                    self._state = self._S_SYNC0
        elif s == self._S_PAYLOAD:
            self._buf.append(b)
            # Telemetry payload now has 3 bytes per sample
            expected_bytes = self._count * 3 if self._type == PKT_TYPE_TELEMETRY else self._count * 2
            if len(self._buf) == expected_bytes: self._state = self._S_CRC
            
        elif s == self._S_CRC:
            if b == crc8(bytes(self._hdr_buf) + bytes(self._buf)):
                if self._type == PKT_TYPE_TELEMETRY:
                    # 'u1' = uint8 (ADC), '<i2' = int16 (Error)
                    dt = np.dtype([('adc', 'u1'), ('err', '<i2')])
                    data = np.frombuffer(self._buf, dtype=dt)
                    yield (self._type, self._seq, data['adc'].astype(np.float32), data['err'].astype(np.float32))
                elif self._type == PKT_TYPE_ADC:
                    samples = np.frombuffer(self._buf, dtype='<u2').astype(np.float32)
                    yield (self._type, self._seq, samples, None)
            self._state = self._S_SYNC0