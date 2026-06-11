"""
tuner.py — Gain tuning engine using the Adam optimiser.

Gradient signal
---------------
At each epoch the MCU's (Ref - Fbk) error waveform is collected and
decomposed into per-harmonic complex phasors  E_k  via windowed DFT.

Loss per channel:  L_k = |E_k|²

Wirtinger gradient of L_k w.r.t. conj(Kp):
    ∂L_k / ∂conj(Kp) ≈ −|E_k|² · conj(G_k)   (closed-loop, G_k = plant)

For a minimum-phase plant the dominant sign is negative, so the
steepest-descent direction in Kp-space is  +conj(E_k).  Adam is applied
with that gradient signal; Kp and Ki each get independent moment
accumulators with per-component (real / imag) second moments so the
adaptive step rate is correctly shaped along both axes.
"""

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import FUND_FREQ, SYS_MODE_STOP, SYS_MODE_RUN, ADC_VREF


# ══════════════════════════════════════════════════════════════════════════════
# Adam state container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _AdamState:
    """
    Per-harmonic Adam optimiser state.

    First moments  (m)  are complex — they track gradient direction.
    Second moments (v)  are real   — they track per-component gradient
    variance; .real for the real axis, .imag for the imaginary axis.
    """

    # ── Kp ────────────────────────────────────────────────────────────────
    m_kp: complex = 0 + 0j   # 1st moment  (exponential mean of gradient)
    v_kp: complex = 0 + 0j   # 2nd moment  (.real = var of Kp_r, .imag = var of Kp_i)

    # ── Ki ────────────────────────────────────────────────────────────────
    m_ki: complex = 0 + 0j
    v_ki: complex = 0 + 0j

    # ── Step counter (bias correction) ───────────────────────────────────
    t: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Tuner engine
# ══════════════════════════════════════════════════════════════════════════════

class TunerEngine:

    # Hardware safety limits
    KP_LIMIT: float = 10.0
    KI_LIMIT: float = 500.0

    # ADC saturation guard — skip the Adam update for any epoch where the
    # ADC waveform is within SAT_MARGIN of either rail.  Clipped samples
    # produce distorted error phasors that would push gains in the wrong
    # direction and cause the divergence shown in the second screenshot.
    SAT_MARGIN: float = 0.05   # 5 % of full-scale  ≈ 165 mV @ 3.3 V

    # Adam hyper-parameters
    BETA1: float = 0.9
    BETA2: float = 0.999
    EPS:   float = 1e-8

    # First-run random initialisation range (±).
    # Small enough to avoid destabilising the plant on first contact;
    # nonzero so Adam has a direction to move from the very first epoch.
    KP_INIT: float = 0.01
    KI_INIT: float = 0.10

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(self,
                 collect_fn:      Callable,
                 discard_fn:      Callable,
                 set_mode_fn:     Callable,
                 set_harmonic_fn: Callable,
                 get_rate_fn:     Callable,
                 on_status_fn:    Callable):

        self._collect      = collect_fn
        self._discard      = discard_fn
        self._set_mode     = set_mode_fn
        self._set_harmonic = set_harmonic_fn
        self._get_rate     = get_rate_fn
        self._on_status    = on_status_fn

        self.running = False
        self._thread: threading.Thread | None = None

        # Gains actively sent to the MCU  {order: {'kp': complex, 'ki': complex}}
        self.gains:    dict[int, dict]       = {}
        self.conv_log: list[tuple]           = []

        # Per-harmonic Adam state
        self._adam: dict[int, _AdamState] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def start(self, orders: list[int], config: dict) -> None:
        if self.running:
            return
        self._init_state(orders)
        self.conv_log.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, args=(orders, config), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        self._set_mode(SYS_MODE_RUN)

    def send_all_gains(self, orders: list[int]) -> None:
        for o in orders:
            self._set_harmonic(o, self.gains[o]['kp'], self.gains[o]['ki'])
            time.sleep(0.005)

    def start_analytical(self, orders: list[int], kp_base: float, ki_base: float) -> None:
        """Starts the analytical tuning sequence in a background GUI-safe thread."""
        if self.running:
            return
        self._init_state(orders)
        self.conv_log.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self.apply_analytical_rule, 
            args=(orders, kp_base, ki_base), 
            daemon=True
        )
        self._thread.start()

    # ── State initialisation ──────────────────────────────────────────────

    # ── Analytical System Identification & Tuning ─────────────────────────

    def identify_plant_response(self, orders: list[int], test_gain: float = 0.05) -> dict[int, complex]:
        """
        Automated empirical measurement of G(jhω) for each harmonic.
        Injects a known test gain and measures the closed-loop complex response.
        """
        fs = self._get_rate()
        spp = max(1, int(fs / FUND_FREQ))
        epoch_n = 10 * spp
        settle_n = 15 * spp
        
        self._on_status({'status': "SysId: Measuring open-loop disturbance..."})
        
        # 1. Zero all gains to measure pure open-loop disturbance (E_open)
        for o in orders:
            self.gains[o] = {'kp': complex(0,0), 'ki': complex(0,0)}
        self.send_all_gains(orders)
        self._discard(settle_n)
        
        adc, err = self._collect(epoch_n, 15.0)
        e_open = self.compute_error_phasors(err, fs, orders)

        g_plant: dict[int, complex] = {}

        # 2. Ping each harmonic sequentially to find G_k
        for o in orders:
            self._on_status({'status': f"SysId: Pinging Harmonic {o}..."})
            
            # Apply test perturbation
            self.gains[o]['kp'] = complex(test_gain, 0)
            self.send_all_gains([o])
            self._discard(settle_n)
            
            adc, err = self._collect(epoch_n, 15.0)
            e_closed = self.compute_error_phasors(err, fs, [o])[o]
            
            # 3. Calculate G(jhω) = (E_open - E_closed) / (K_test * E_closed)
            denom = test_gain * e_closed
            if abs(denom) < 1e-9:
                g_plant[o] = complex(1e-9, 0)  # Failsafe for zero division
            else:
                g_plant[o] = (e_open[o] - e_closed) / denom
                
            # Reset this harmonic before testing the next one
            self.gains[o]['kp'] = complex(0,0)
            self.send_all_gains([o])

        return g_plant

    def apply_analytical_rule(self, orders: list[int], kp_base: float, ki_base: float) -> None:
        """
        Executes the tuning rule: K(h) = k / G(jhω)
        Treats kp_base and ki_base as the optimal ITAE scalars.
        """
        self.running = True
        self._set_mode(SYS_MODE_RUN)
        
        try:
            # Step 1: Identify the hardware
            g_plant = self.identify_plant_response(orders)
            
            # Step 2: Apply the analytical inversion
            for o in orders:
                G = g_plant[o]
                
                # Apply: Kp(h) = kp / G(jhω)
                self.gains[o]['kp'] = kp_base / G
                
                # Apply: Ki(h) = ki / G(jhω)
                self.gains[o]['ki'] = ki_base / G

            self._on_status({'status': "Applying Analytical Gains..."})
            self.send_all_gains(orders)
            
            # Measure final loss for UI logging
            self._discard(20 * int(self._get_rate() / FUND_FREQ))
            adc, err = self._collect(10 * int(self._get_rate() / FUND_FREQ), 15.0)
            loss, thd = self.compute_loss(err, adc, self._get_rate(), use_thd=False)
            
            self._on_status({
                'status': 'Analytical Tuning Complete',
                'loss': loss,
                'gains': self.gains,
                'done': True
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._on_status({'status': f"SysId Error: {e}", 'done': True})
            
        self.running = False
    

    def _init_state(self, orders: list[int]) -> None:
        """
        Prepare state for a new optimiser run.

        Gains
        -----
        Preserved across runs so the optimiser continues from where it
        left off.  An order seen for the first time gets a small random
        starting point instead of zero — this breaks the symmetry that
        keeps Adam stationary at the origin on the first epoch.

        Adam state
        ----------
        Always reset.  Momentum and variance accumulators reflect the old
        trajectory and would push gains in a stale direction if carried
        over into a fresh run.
        """
        rng = np.random.default_rng()

        for o in orders:
            if o not in self.gains:
                # First time this harmonic is used — randomise within a
                # small safe range so the plant is not immediately disturbed.
                self.gains[o] = {
                    'kp': complex(
                        rng.uniform(-self.KP_INIT, self.KP_INIT),
                        rng.uniform(-self.KP_INIT, self.KP_INIT),
                    ),
                    'ki': complex(
                        rng.uniform(-self.KI_INIT, self.KI_INIT),
                        rng.uniform(-self.KI_INIT, self.KI_INIT),
                    ),
                }
            # Adam moments always start fresh — new run, clean slate.
            self._adam[o] = _AdamState()

    # ── Signal processing ─────────────────────────────────────────────────

    @staticmethod
    def extract_phasor(signal: np.ndarray, fs: float, freq: float) -> complex:
        """
        Extract the complex phasor at *freq* Hz from *signal* using a
        Hanning-windowed correlator.  Returns complex(mean, 0) for DC.
        """
        n = len(signal)
        if freq < 1e-6:
            return complex(float(np.mean(signal)), 0.0)

        t      = np.arange(n) / fs
        omega  = 2.0 * np.pi * freq
        window = np.hanning(n)
        w_sig  = signal * window * 2.0          # ×2 compensates Hanning amplitude loss

        r =  float(np.dot(w_sig, np.cos(omega * t))) * 2.0 / n
        i =  float(np.dot(w_sig, np.sin(omega * t))) * 2.0 / n
        return complex(r, -i)                   # convention: pure-sin → complex(0, -A)

    def compute_error_phasors(self,
                              err:    np.ndarray,
                              fs:     float,
                              orders: list[int]) -> dict[int, complex]:
        """
        Decompose the MCU error waveform into one complex phasor per harmonic.
        Driving every  E_k → 0  is the sole optimisation objective.
        """
        return {
            o: self.extract_phasor(err, fs, o * FUND_FREQ)
            for o in orders
        }

    def compute_loss(self,
                     err:     np.ndarray,
                     adc:     np.ndarray,
                     fs:      float,
                     use_thd: bool) -> tuple[float, float]:
        """
        Scalar loss and THD% for convergence logging.
        THD requires the full ADC waveform; the primary loss uses RMS error.
        """
        sig_ac = adc - float(np.mean(adc))
        h1     = self.extract_phasor(sig_ac, fs, FUND_FREQ)
        v1     = abs(h1)

        thd_pct = 0.0
        if v1 > 0.01:
            harm_power = sum(
                abs(self.extract_phasor(sig_ac, fs, k * FUND_FREQ)) ** 2
                for k in range(2, 11)
            )
            thd_pct = 100.0 * np.sqrt(harm_power) / v1

        loss = (thd_pct / 100.0
                if (use_thd and v1 > 0.01)
                else float(np.sqrt(np.mean(err ** 2))))

        return loss, thd_pct

    # ── Adam optimiser ────────────────────────────────────────────────────

    def _adam_step(self,
                   error_phasors: dict[int, complex],
                   orders:        list[int],
                   lr_kp:         float,
                   lr_ki:         float) -> None:
        """
        One Adam update for every active harmonic channel.

        Gradient
        --------
        g = conj(E_k)

        The real and imaginary parts of g are treated as independent
        scalar parameters, each with its own second-moment accumulator,
        so the adaptive learning rate is shaped per axis.

        Update direction
        ----------------
        gains += lr · w_k · step        (note: + not −)

        The − sign from gradient descent is absorbed by the choice of
        g = conj(E_k) as the *descent* direction (not the gradient itself),
        which holds for minimum-phase plants where reducing |E_k| requires
        increasing the gain magnitude along E_k's phase angle.

        Per-harmonic weighting
        ----------------------
        Adam's second-moment normalisation drives every channel's step to
        approximately ±1 regardless of  |E_k|.  Without an explicit weight,
        a harmonic contributing 0.01 % of total error gets the same step
        magnitude as the dominant harmonic.

        The weight  w_k = |E_k| / total_rms  is the gradient of the RMS
        loss  L = √(Σ|E_k|²)  w.r.t. the per-channel loss  |E_k|².
        Applying it *after* Adam's normalisation preserves Adam's noise
        rejection and direction adaptation while correctly allocating
        learning effort in proportion to each harmonic's share of the
        total error.
        """
        b1, b2, eps = self.BETA1, self.BETA2, self.EPS

        # ── Per-harmonic weights from RMS loss gradient ───────────────────
        total_rms = np.sqrt(sum(abs(error_phasors[o]) ** 2
                                for o in orders)) + eps

        for o in orders:
            w = abs(error_phasors[o]) / total_rms   # ∈ (0, 1], sums in quadrature to 1
            E  = error_phasors[o]
            
            # --- 1. DECOUPLED GRADIENT CALCULATION ---
            g_kp = complex(E.real, -E.imag)         # conj(E_k) — Kp descent direction
            
            if o > 0:
                omega = 2.0 * np.pi * (o * FUND_FREQ)
                # Rotate Ki descent direction by -90 degrees (-j) and scale by omega
                g_ki = g_kp * complex(0, -omega)
            else:
                g_ki = g_kp
            # -----------------------------------------
            
            s  = self._adam[o]

            s.t += 1
            t   = s.t

            # ── 1st moments (complex — track direction) ───────────────────
            # Update Kp and Ki using their respective decoupled gradients
            s.m_kp = b1 * s.m_kp + (1.0 - b1) * g_kp
            s.m_ki = b1 * s.m_ki + (1.0 - b1) * g_ki

            # ── 2nd moments (per-component — track variance per axis) ─────
            # Calculate the squared magnitude for variance separately
            g_sq_kp = complex(g_kp.real * g_kp.real, g_kp.imag * g_kp.imag)
            g_sq_ki = complex(g_ki.real * g_ki.real, g_ki.imag * g_ki.imag)
            
            s.v_kp = b2 * s.v_kp + (1.0 - b2) * g_sq_kp
            s.v_ki = b2 * s.v_ki + (1.0 - b2) * g_sq_ki

            # ── Bias correction ───────────────────────────────────────────
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
    # ── Main tuning loop ──────────────────────────────────────────────────

    def _loop(self, orders: list[int], cfg: dict) -> None:
        try:
            fs        = self._get_rate()
            spp       = max(1, int(fs / FUND_FREQ))
            epoch_n   = max(5, cfg['n_periods']) * spp
            discard_n = 20 * spp
            settle_n  = 10 * spp
            lr_kp     = cfg['lr_kp']
            lr_ki     = cfg['lr_ki']

            for epoch in range(cfg['max_epochs']):
                if not self.running:
                    break

                self._on_status({
                    'status': f"Epoch {epoch + 1}/{cfg['max_epochs']}",
                    'epoch':  epoch + 1,
                })

                # ── Mode 1: stop → flush → restart → settle ───────────────
                if cfg['mode'] == 'Mode1':
                    self._set_mode(SYS_MODE_STOP)
                    time.sleep(0.05)
                    self._collect(epoch_n, 15.0)        # flush stale data
                    self._set_mode(SYS_MODE_RUN)
                    self._discard(settle_n)

                # ── Collect measurement window ────────────────────────────
                adc, err = self._collect(epoch_n, 15.0)
                if adc is None or err is None:
                    break

                fs = self._get_rate()

                # ── Saturation guard ──────────────────────────────────────
                # If the ADC is railing, the error phasors are computed from
                # a clipped waveform.  The resulting gradients are corrupted
                # and will drive gains further into saturation.  Skip the
                # Adam update entirely for this epoch and log a warning.
                lo = ADC_VREF *      self.SAT_MARGIN
                hi = ADC_VREF * (1 - self.SAT_MARGIN)
                adc_min, adc_max = float(adc.min()), float(adc.max())
                if adc_min < lo or adc_max > hi:
                    self._on_status({'status': "ADC saturated — backing out!"})
                    # Forcefully shrink the gains by 5% to escape the saturation rail
                    for o in orders:
                        self.gains[o]['kp'] *= 0.90
                        self.gains[o]['ki'] *= 0.90
                    self.send_all_gains(orders)
                    continue

                # ── Gradient signal: per-harmonic error phasors ───────────
                error_phasors = self.compute_error_phasors(err, fs, orders)

                # ── Scalar loss for convergence logging ───────────────────
                loss, thd = self.compute_loss(err, adc, fs, cfg['use_thd'])

                # ── Adam update → send to MCU ─────────────────────────────
                self._adam_step(error_phasors, orders, lr_kp, lr_ki)
                self.send_all_gains(orders)

                # ── Mode 2: discard post-update transient ─────────────────
                if cfg['mode'] == 'Mode2':
                    self._discard(discard_n)

                # ── Log ───────────────────────────────────────────────────
                self.conv_log.append((epoch + 1, loss, thd))
                self._on_status({
                    'status': f"Epoch {epoch + 1}/{cfg['max_epochs']}",
                    'epoch':  epoch + 1,
                    'loss':   loss,
                    'thd':    thd,
                    'gains':  self.gains,
                })

            self._set_mode(SYS_MODE_RUN)
            self._on_status({'status': 'Converged', 'done': True})

        except Exception as e:
            traceback.print_exc()
            self._on_status({'status': f"Error: {e}", 'done': True})