"""
transport.py — Serial connection, reader thread, command sender.

SerialManager owns:
  - The serial port object
  - The reader thread
  - sample_buf (deque) and stats (dict) shared with the UI
  - The tuner collection buffer (_tbuf) shared with TunerEngine
"""
import collections
import struct
import threading
import time
import numpy as np
import serial
import serial.tools.list_ports

from .config import (ADC_VREF, ADC_MAX_CODE, DISPLAY_LEN,
                     PKT_TYPE_ADC, PKT_TYPE_CMD, PKT_TYPE_TELEMETRY,
                     CMD_SET_MODE, CMD_SET_HARMONIC,
                     SAMPLE_RATE_NOM)
from .protocol import PacketParser, crc8

class SerialManager:
    STM32_VID = 0x0483

    def __init__(self):
        self._serial: serial.Serial | None = None
        self._reader_thread: threading.Thread | None = None
        self.connected = False
        self._lock = threading.Lock()

        self.sample_buf = collections.deque(maxlen=DISPLAY_LEN)
        self.error_buf  = collections.deque(maxlen=DISPLAY_LEN)
        self.stats = dict(total_samples=0, total_packets=0, dropped=0,
                          last_seq=None, rate_hz=0.0,
                          min_v=0.0, max_v=0.0, mean_v=0.0)

        self._tbuf_adc: list[float] = []
        self._tbuf_err: list[float] = []
        self._tbuf_lock  = threading.Lock()
        self._tbuf_event = threading.Event()
        self._tbuf_active = False

        self.on_packet: callable | None = None
        self.on_disconnect: callable | None = None

    def list_ports(self) -> tuple[list[str], dict[str, str]]:
        stm, other = [], []
        port_map: dict[str, str] = {}
        for p in sorted(serial.tools.list_ports.comports()):
            desc = (p.description or "Unknown").replace(p.device, "").strip(" -()")
            label = f"{p.device}  -  {desc}" if desc else p.device
            if getattr(p, 'vid', None) == self.STM32_VID:
                label = f"[STM32] {label}"; stm.append(label)
            else:
                other.append(label)
            port_map[label] = p.device
        return stm + other, port_map

    def connect(self, port: str, baud: int) -> bool:
        try:
            self._serial = serial.Serial(port, baud, timeout=1.0)
        except serial.SerialException:
            return False
        self.connected = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        return True

    def disconnect(self):
        self.connected = False
        self.stop_collection()
        if self._serial:
            try: self._serial.close()
            except: pass
            self._serial = None

    def start_collection(self):
        self._tbuf_adc.clear()
        self._tbuf_err.clear()
        self._tbuf_active = True

    def stop_collection(self):
        self._tbuf_active = False
        self._tbuf_event.set()

    def collect_samples(self, n: int, timeout: float = 15.0) -> tuple[np.ndarray | None, np.ndarray | None]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._tbuf_lock:
                if len(self._tbuf_adc) >= n:
                    out_a = np.array(self._tbuf_adc[:n])
                    out_e = np.array(self._tbuf_err[:n])
                    self._tbuf_adc = self._tbuf_adc[n:]
                    self._tbuf_err = self._tbuf_err[n:]
                    return out_a, out_e
            self._tbuf_event.wait(timeout=0.05)
            self._tbuf_event.clear()
        return None, None

    def discard_samples(self, n: int, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        remaining = n
        while time.monotonic() < deadline and remaining > 0:
            with self._tbuf_lock:
                take = min(len(self._tbuf_adc), remaining)
                if take:
                    self._tbuf_adc = self._tbuf_adc[take:]
                    self._tbuf_err = self._tbuf_err[take:]
                    remaining -= take
            if remaining > 0:
                self._tbuf_event.wait(timeout=0.05)
                self._tbuf_event.clear()

    def send_command(self, cmd_id: int, payload: bytes) -> bool:
        if not self.connected or not self._serial: return False
        type_bytes = struct.pack('<H', PKT_TYPE_CMD)
        cmd_byte   = bytes([cmd_id])
        crc_data   = type_bytes + cmd_byte + payload
        frame      = b'\xaa\x55' + crc_data + bytes([crc8(crc_data)])
        try:
            self._serial.write(frame)
            return True
        except Exception:
            return False

    def cmd_set_mode(self, mode: int) -> bool:
        return self.send_command(CMD_SET_MODE, bytes([mode]))

    def cmd_set_harmonic(self, order: int, kp: complex, ki: complex) -> bool:
        payload = bytes([order]) + struct.pack('<4f', kp.real, kp.imag, ki.real, ki.imag)
        return self.send_command(CMD_SET_HARMONIC, payload)

    def _reader_loop(self):
        parser = PacketParser()
        t0 = time.monotonic(); count = 0

        while self.connected and self._serial:
            try:
                if not getattr(self._serial, 'is_open', False): break
                raw = self._serial.read(512)
            except (serial.SerialException, TypeError, AttributeError, OSError):
                if self.on_disconnect: self.on_disconnect()
                break
            if not raw: continue

            for pkt_type, seq, adc_raw, err_raw in parser.feed(raw):
                if pkt_type not in (PKT_TYPE_ADC, PKT_TYPE_TELEMETRY): continue

                with self._lock:
                    s = self.stats
                    if s["last_seq"] is not None:
                        exp = (s["last_seq"] + 1) & 0xFFFFFFFF
                        if seq != exp:
                            s["dropped"] += int(seq - exp) & 0xFFFFFFFF
                    s["last_seq"] = seq
                    s["total_packets"] += 1
                    s["total_samples"] += len(adc_raw)

                
                if pkt_type == PKT_TYPE_TELEMETRY:
                    volts = adc_raw * (ADC_VREF / 255.0) # 8-bit scaling
                else:
                    volts = adc_raw * (ADC_VREF / ADC_MAX_CODE) # 12-bit scaling
                    
                self.sample_buf.extend(volts)
                
                if err_raw is not None:
                    # Error is still calculated in 12-bit LSBs on the MCU hardware
                    err_volts = err_raw * (ADC_VREF / ADC_MAX_CODE)
                    self.error_buf.extend(err_volts)

                if self._tbuf_active:
                    with self._tbuf_lock:
                        self._tbuf_adc.extend(volts.tolist())
                        if err_raw is not None:
                            self._tbuf_err.extend(err_volts.tolist())
                    self._tbuf_event.set()

                if self.on_packet: self.on_packet(volts)

                count += len(adc_raw)
                elapsed = time.monotonic() - t0
                if elapsed >= 1.0:
                    with self._lock:
                        self.stats["rate_hz"] = count / elapsed
                    count = 0; t0 = time.monotonic()

                if self.sample_buf:
                    arr = np.array(self.sample_buf)
                    with self._lock:
                        self.stats.update(min_v=float(arr.min()), max_v=float(arr.max()), mean_v=float(arr.mean()))

                

    def get_rate(self) -> float:
        with self._lock:
            return self.stats.get("rate_hz", SAMPLE_RATE_NOM) or SAMPLE_RATE_NOM