"""
config.py — Shared constants and signal parameters.
All values must match the firmware configuration.
"""

# ── ADC / DAC ─────────────────────────────────────────────────────────────────
ADC_VREF     = 3.3
ADC_MAX_CODE = 4095
DISPLAY_LEN  = 4096

# ── Serial ────────────────────────────────────────────────────────────────────
BAUD_RATES = ["9600","19200","38400","57600","115200",
              "230400","460800","921600"]

# ── Protocol ─────────────────────────────────────────────────────────────────
PKT_TYPE_ADC       = 0x0001
PKT_TYPE_CMD       = 0x0002
PKT_TYPE_TELEMETRY = 0x0003
CMD_SET_MODE     = 0x01
CMD_SET_HARMONIC = 0x02
SYS_MODE_STOP    = 0
SYS_MODE_RUN     = 1

# ── Signal parameters (must match firmware) ───────────────────────────────────
FUND_FREQ       = 50.0
SAMPLE_RATE_NOM = 20000.0
SPP             = int(SAMPLE_RATE_NOM / FUND_FREQ)   # samples per period

# Reference phasors in volts.
# extract_phasor() convention: returns complex(0, -A) for a pure sin of amplitude A.
_REF_H1_AMP = 0.8 * 1000.0 * (ADC_VREF / ADC_MAX_CODE)   # ~0.645 V
REFERENCE_PHASORS = {
    0: complex(0.0,        0.0),
    1: complex(0.0, -_REF_H1_AMP),
}
