"""
app.py — MonitorApp: main window, plots, UI wiring.

MonitorApp owns the SerialManager and TunerEngine and wires them together
through callbacks. It never calls serial or numpy directly — that stays
in transport.py and tuner.py.
"""

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np

from .config import (ADC_VREF, ADC_MAX_CODE, DISPLAY_LEN, BAUD_RATES,
                     FUND_FREQ, SAMPLE_RATE_NOM, SYS_MODE_STOP, SYS_MODE_RUN)
from .transport import SerialManager
from .tuner     import TunerEngine
from .dialogs   import SetGainsDialog, save_gains_to_file


class MonitorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("STM32F429 Harmonic Control Arrays Real Time Gain Optimizer")
        self.root.minsize(960, 640)
        self.root.configure(bg="#1e1e2e")

        # ── Core objects ───────────────────────────────────────────────────────
        self._transport = SerialManager()
        self._transport.on_disconnect = lambda: self.root.after(0, self._disconnect)

        self._tuner = TunerEngine(
            collect_fn      = self._transport.collect_samples,
            discard_fn      = self._transport.discard_samples,
            set_mode_fn     = self._transport.cmd_set_mode,
            set_harmonic_fn = self._transport.cmd_set_harmonic,
            get_rate_fn     = self._transport.get_rate,
            on_status_fn    = lambda d: self.root.after(0, lambda: self._on_tuner_status(d)),
        )
        

        self._port_map: dict[str, str] = {}

        self._halted = False

        self._build_ui()
        self._refresh_ports()
        self._schedule_update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self._build_conn_bar()
        self._build_tuner_bar()
        self._build_analytical_bar() # <--- Add this line
        self._build_status_bar()
        self._build_plots()
        self._build_stats_bar()

    def _build_analytical_bar(self):
        bar = tk.Frame(self.root, bg="#2a2a3e", pady=5, padx=10)
        bar.pack(fill=tk.X, side=tk.TOP)
        L = dict(bg="#2a2a3e", fg="#cba6f7", font=("Segoe UI", 9, "bold"))

        tk.Label(bar, text="Analytical SysId", **L).pack(side=tk.LEFT, padx=(0, 10))
        
        # Base Gain Inputs
        tk.Label(bar, text="Base Kp:", bg="#2a2a3e", fg="#cdd6f4", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 3))
        self._base_kp_var = tk.StringVar(value="0.5")
        tk.Entry(bar, textvariable=self._base_kp_var, width=6, bg="#313244", fg="#cdd6f4", relief=tk.FLAT, font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(bar, text="Base Ki:", bg="#2a2a3e", fg="#cdd6f4", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 3))
        self._base_ki_var = tk.StringVar(value="2.0")
        tk.Entry(bar, textvariable=self._base_ki_var, width=6, bg="#313244", fg="#cdd6f4", relief=tk.FLAT, font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 16))

        # Trigger Button
        self._btn_analytical = tk.Button(
            bar, text="Run Plant Inversion", width=18, bg="#cba6f7", fg="#1e1e2e",
            relief=tk.FLAT, font=("Segoe UI", 9, "bold"), cursor="hand2",
            command=self._start_analytical_tuning
        )
        self._btn_analytical.pack(side=tk.LEFT)

    def _start_analytical_tuning(self):
        if self._tuner.running or not self._transport.connected: 
            return
            
        try:
            orders = [int(x.strip()) for x in self._harmonics_var.get().split(",")]
            kp_base = float(self._base_kp_var.get())
            ki_base = float(self._base_ki_var.get())
        except ValueError:
            self._tv_status.set("Bad inputs")
            return

        self._save_btn.configure(state=tk.NORMAL)
        self._rebuild_gain_labels(orders)

        # Prepare hardware buffer and start the background thread
        self._transport.start_collection()
        self._tuner.start_analytical(orders, kp_base, ki_base)
        
        # Lock UI buttons while running
        self._start_btn.configure(state=tk.DISABLED)
        self._btn_analytical.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL, bg="#f38ba8", fg="#1e1e2e")

    def _build_conn_bar(self):
        bar = tk.Frame(self.root, bg="#313244", pady=6, padx=10)
        bar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(bar, text="Port", bg="#313244", fg="#cdd6f4",
                 font=("Segoe UI",9)).pack(side=tk.LEFT, padx=(0,4))
        self._port_var = tk.StringVar()
        self._port_cb  = ttk.Combobox(bar, textvariable=self._port_var,
                                      width=38, state="readonly")
        self._port_cb.pack(side=tk.LEFT, padx=(0,4))
        self._port_count_var = tk.StringVar(value="")
        tk.Button(bar, text="<<", bg="#45475a", fg="#cdd6f4", relief=tk.FLAT,
                  cursor="hand2", command=self._refresh_ports
                  ).pack(side=tk.LEFT, padx=(0,4))
        tk.Label(bar, textvariable=self._port_count_var, bg="#313244",
                 fg="#6c7086", font=("Segoe UI",8)).pack(side=tk.LEFT, padx=(0,12))

        tk.Label(bar, text="Baud", bg="#313244", fg="#cdd6f4",
                 font=("Segoe UI",9)).pack(side=tk.LEFT, padx=(0,4))
        self._baud_var = tk.StringVar(value="921600")
        ttk.Combobox(bar, textvariable=self._baud_var, values=BAUD_RATES,
                     width=9, state="readonly").pack(side=tk.LEFT, padx=(0,16))

        self._conn_btn = tk.Button(bar, text="Connect", width=10,
                                   bg="#a6e3a1", fg="#1e1e2e", relief=tk.FLAT,
                                   font=("Segoe UI",9,"bold"), cursor="hand2",
                                   command=self._toggle_connection)
        self._conn_btn.pack(side=tk.LEFT, padx=(0,10))

        self._status_canvas = tk.Canvas(bar, width=12, height=12,
                                        bg="#313244", highlightthickness=0)
        self._status_canvas.pack(side=tk.LEFT, padx=(0,4))
        self._status_dot = self._status_canvas.create_oval(1,1,11,11,
                                                           fill="#585b70", outline="")
        self._status_lbl = tk.Label(bar, text="Disconnected", bg="#313244",
                                    fg="#585b70", font=("Segoe UI",9))
        self._status_lbl.pack(side=tk.LEFT)

        self._halt_btn = tk.Button(bar, text="Halt", width=6, bg="#f9e2af", fg="#1e1e2e",
                           relief=tk.FLAT, font=("Segoe UI",9,"bold"), cursor="hand2",
                           command=self._halt)
        self._halt_btn.pack(side=tk.LEFT, padx=(10,0))

    def _build_tuner_bar(self):
        bar = tk.Frame(self.root, bg="#2a2a3e", pady=5, padx=10)
        bar.pack(fill=tk.X, side=tk.TOP)
        L = dict(bg="#2a2a3e", fg="#cdd6f4", font=("Segoe UI",9))

        tk.Label(bar, text="Auto-Tune", **L).pack(side=tk.LEFT, padx=(0,10))
        tk.Label(bar, text="Mode:", **L).pack(side=tk.LEFT, padx=(0,4))
        self._mode_var = tk.StringVar(value="Mode1")
        for text, val in [("Mode 1","Mode1"),("Mode 2","Mode2")]:
            tk.Radiobutton(bar, text=text, variable=self._mode_var, value=val,
                           bg="#2a2a3e", fg="#cdd6f4", selectcolor="#45475a",
                           activebackground="#2a2a3e", font=("Segoe UI",8)
                           ).pack(side=tk.LEFT, padx=(0,6))

        tk.Label(bar, text="|", bg="#2a2a3e", fg="#585b70").pack(side=tk.LEFT, padx=4)

        def entry(label, var, width=5):
            tk.Label(bar, text=label, **L).pack(side=tk.LEFT, padx=(0,3))
            tk.Entry(bar, textvariable=var, width=width,
                     bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                     relief=tk.FLAT, font=("Consolas",9)).pack(side=tk.LEFT, padx=(0,8))

        self._n_periods_var  = tk.StringVar(value="20");  entry("N periods:", self._n_periods_var)
        self._harmonics_var  = tk.StringVar(value="0,1");  entry("Harmonics:", self._harmonics_var, 10)
        self._thd_var        = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="THD", variable=self._thd_var,
                       bg="#2a2a3e", fg="#cdd6f4", selectcolor="#45475a",
                       activebackground="#2a2a3e", font=("Segoe UI",8)
                       ).pack(side=tk.LEFT, padx=(0,8))
        self._lr_kp_var      = tk.StringVar(value="0.05"); entry("LR Kp:", self._lr_kp_var, 6)
        self._lr_ki_var      = tk.StringVar(value="0.2");  entry("LR Ki:", self._lr_ki_var, 6)
        self._max_epochs_var = tk.StringVar(value="100");  entry("Epochs:", self._max_epochs_var)

        self._start_btn = tk.Button(bar, text="Start", width=7, bg="#89b4fa",
                                    fg="#1e1e2e", relief=tk.FLAT,
                                    font=("Segoe UI",9,"bold"), cursor="hand2",
                                    command=self._start_tuning)
        self._start_btn.pack(side=tk.LEFT, padx=(0,4))
        self._stop_btn = tk.Button(bar, text="Stop", width=7, bg="#45475a",
                                   fg="#cdd6f4", relief=tk.FLAT,
                                   font=("Segoe UI",9), cursor="hand2",
                                   command=self._stop_tuning, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=(0,8))

        tk.Label(bar, text="|", bg="#2a2a3e", fg="#585b70").pack(side=tk.LEFT, padx=4)
        self._save_btn = tk.Button(bar, text="Save Gains", width=9, bg="#a6e3a1",
                                   fg="#1e1e2e", relief=tk.FLAT,
                                   font=("Segoe UI",9), cursor="hand2",
                                   command=self._save_gains, state=tk.DISABLED)
        self._save_btn.pack(side=tk.LEFT, padx=(0,4))
        tk.Button(bar, text="Set Gains", width=9, bg="#f9e2af", fg="#1e1e2e",
                  relief=tk.FLAT, font=("Segoe UI",9), cursor="hand2",
                  command=self._open_set_gains).pack(side=tk.LEFT)

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg="#181825", pady=3, padx=10)
        bar.pack(fill=tk.X, side=tk.TOP)

        SL = dict(bg="#181825", fg="#6c7086", font=("Segoe UI",8))
        SV = dict(bg="#181825", fg="#cdd6f4", font=("Consolas",8,"bold"))
        self._tv_status = tk.StringVar(value="Idle")
        self._tv_epoch  = tk.StringVar(value="--")
        self._tv_loss   = tk.StringVar(value="--")
        self._tv_thd    = tk.StringVar(value="--")

        for lbl, var, w in [("Status:", self._tv_status, 14),
                             ("Epoch:",  self._tv_epoch,  10),
                             ("Loss:",   self._tv_loss,   12),
                             ("THD:",    self._tv_thd,    10)]:
            tk.Label(bar, text=lbl, **SL).pack(side=tk.LEFT)
            tk.Label(bar, textvariable=var, width=w, **SV).pack(
                     side=tk.LEFT, padx=(0,16))

        self._gains_frame = tk.Frame(bar, bg="#181825")
        self._gains_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._gain_labels: dict[int, tk.StringVar] = {}

    def _build_plots(self):
        
        frame = tk.Frame(self.root, bg="#1e1e2e")
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4,0))

        fig = Figure(facecolor="#1e1e2e")
        fig.subplots_adjust(hspace=0.6, top=0.93, bottom=0.08, left=0.07, right=0.97)
        self._fig = fig

        self._ax_wave = fig.add_subplot(4,1,1)
        self._ax_fft  = fig.add_subplot(4,1,2)
        self._ax_err  = fig.add_subplot(4,1,3)
        self._ax_conv = fig.add_subplot(4,1,4)
        self._style_axes()

        self._wave_line, = self._ax_wave.plot([], [], lw=0.8, color="#89b4fa")
        self._fft_line,  = self._ax_fft.plot([],  [], lw=1.0, color="#cba6f7")
        self._err_line,  = self._ax_err.plot([],  [], lw=0.8, color="#f38ba8")
        self._conv_line, = self._ax_conv.plot([], [], lw=1.2, color="#a6e3a1", marker=".", ms=4)
        
        self._harmonic_vlines = [
            self._ax_fft.axvline(x=0, color="#f38ba8", lw=0.6, alpha=0.5, ls="--")
            for _ in range(10)
        ]
        self._harmonic_annotations: list = []

        self._canvas = FigureCanvasTkAgg(fig, master=frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_stats_bar(self):
        bar = tk.Frame(self.root, bg="#0f0f1a", pady=3, padx=10)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        SL = dict(bg="#0f0f1a", fg="#a6adc8", font=("Segoe UI",8))
        SV = dict(bg="#0f0f1a", fg="#cdd6f4", font=("Consolas",8,"bold"))

        def stat(label):
            tk.Label(bar, text=label, **SL).pack(side=tk.LEFT)
            v = tk.StringVar(value="---")
            tk.Label(bar, textvariable=v, width=12, **SV).pack(
                     side=tk.LEFT, padx=(0,14))
            return v

        self._sv_rate = stat("Sample rate:")
        self._sv_min  = stat("Min:")
        self._sv_max  = stat("Max:")
        self._sv_mean = stat("Mean:")
        self._sv_pkts = stat("Packets:")
        self._sv_drop = stat("Dropped:")

    def _style_axes(self):
        bg = "#1e1e2e"; fg = "#cdd6f4"; grid = "#313244"
        for ax, title, ylabel in [
            (self._ax_wave, "ADC Waveform",             "Voltage (V)"),
            (self._ax_fft,  "Frequency Spectrum (FFT)", "Magnitude (dBV)"),
            (self._ax_err,  "MCU Error (Ref - Fbk)",    "Error (V)"),
            (self._ax_conv, "Convergence",              "Loss"),
        ]:
            ax.set_facecolor(bg); ax.set_title(title, color=fg, fontsize=8, pad=3)
            ax.set_ylabel(ylabel, color=fg, fontsize=7)
            ax.tick_params(colors=fg, labelsize=6)
            for sp in ax.spines.values(): sp.set_color(grid)
            ax.grid(True, color=grid, linewidth=0.5)
            
        self._ax_wave.set_xlim(0, DISPLAY_LEN); self._ax_wave.set_ylim(-0.05, ADC_VREF + 0.05)
        self._ax_err.set_xlim(0, DISPLAY_LEN);  self._ax_err.set_ylim(-1.5, 1.5)
        
        self._ax_fft.set_xlim(0, 500); self._ax_fft.set_ylim(-80, 10)
        self._ax_conv.set_xlim(0, 10); self._ax_conv.set_ylim(0, 1)
    # ══════════════════════════════════════════════════════════════════════════
    # Connection management
    # ══════════════════════════════════════════════════════════════════════════
    def _refresh_ports(self):
        labels, self._port_map = self._transport.list_ports()
        self._port_cb["values"] = labels
        if self._port_var.get() not in labels:
            self._port_var.set(labels[0] if labels else "")
        self._port_count_var.set(
            f"{len(labels)} port{'s' if len(labels)!=1 else ''}")

    def _toggle_connection(self):
        if self._transport.connected: self._disconnect()
        else:                         self._connect()

    def _connect(self):
        label = self._port_var.get()
        port  = self._port_map.get(label, label)
        if not port:
            self._set_conn_status("No port", "#f38ba8", "#f38ba8"); return
        if not self._transport.connect(port, int(self._baud_var.get())):
            self._set_conn_status("Connect failed", "#f38ba8", "#f38ba8"); return
        self._conn_btn.configure(text="Disconnect", bg="#f38ba8")
        self._port_cb.configure(state="disabled")
        self._set_conn_status(f"Connected  {port}", "#a6e3a1", "#a6e3a1")

    def _disconnect(self):
        if self._tuner.running: self._stop_tuning()
        self._halted = False
        self._halt_btn.configure(text="Halt", bg="#f9e2af", fg="#1e1e2e")
        self._transport.disconnect()
        self._conn_btn.configure(text="Connect", bg="#a6e3a1", fg="#1e1e2e")
        self._port_cb.configure(state="readonly")
        self._set_conn_status("Disconnected", "#585b70", "#585b70")
        self._save_btn.configure(state=tk.DISABLED)

    def _set_conn_status(self, text, dot, fg):
        self._status_canvas.itemconfigure(self._status_dot, fill=dot)
        self._status_lbl.configure(text=text, fg=fg)


    def _halt(self):
        if self._halted:
            self._transport.cmd_set_mode(SYS_MODE_RUN)
            self._halt_btn.configure(text="Halt", bg="#f9e2af", fg="#1e1e2e")
            self._halted = False
        else:
            self._transport.cmd_set_mode(SYS_MODE_STOP)
            self._halt_btn.configure(text="Resume", bg="#f38ba8", fg="#1e1e2e")
            self._halted = True



    # ══════════════════════════════════════════════════════════════════════════
    # Tuning
    # ══════════════════════════════════════════════════════════════════════════
    def _start_tuning(self):
        if self._tuner.running or not self._transport.connected: return
        try:
            orders = [int(x.strip()) for x in self._harmonics_var.get().split(",")]
        except ValueError:
            self._tv_status.set("Bad harmonics"); return

        self._save_btn.configure(state=tk.NORMAL)
        self._rebuild_gain_labels(orders)

        self._transport.start_collection()
        self._tuner.start(orders, {
            'mode':       self._mode_var.get(),
            'n_periods':  int(self._n_periods_var.get()),
            'lr_kp':      float(self._lr_kp_var.get()),
            'lr_ki':      float(self._lr_ki_var.get()),
            'max_epochs': int(self._max_epochs_var.get()),
            'use_thd':    self._thd_var.get(),
        })
        self._start_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL, bg="#f38ba8", fg="#1e1e2e")


    def _stop_tuning(self, preserve_error=False):
        self._tuner.stop()
        self._transport.stop_collection()
        
        # Re-enable all start buttons
        self._start_btn.configure(state=tk.NORMAL)
        if hasattr(self, '_btn_analytical'):
            self._btn_analytical.configure(state=tk.NORMAL)
            
        self._stop_btn.configure(state=tk.DISABLED, bg="#45475a", fg="#cdd6f4")
        
        if not preserve_error:
            self._tv_status.set("Stopped")

    

    def _rebuild_gain_labels(self, orders: list[int]):
        for w in self._gains_frame.winfo_children(): w.destroy()
        self._gain_labels.clear()
        for o in orders:
            tk.Label(self._gains_frame, text=f"H{o}:",
                     bg="#181825", fg="#6c7086",
                     font=("Segoe UI",8)).pack(side=tk.LEFT)
            v = tk.StringVar(value="---")
            tk.Label(self._gains_frame, textvariable=v, width=28,
                     bg="#181825", fg="#f9e2af",
                     font=("Consolas",8)).pack(side=tk.LEFT, padx=(0,12))
            self._gain_labels[o] = v

    # 2. Update _on_tuner_status to check for errors before stopping
    def _on_tuner_status(self, d: dict):
        if 'status' in d: self._tv_status.set(d['status'])
        if 'epoch'  in d: self._tv_epoch.set(str(d['epoch']))
        if 'loss'   in d: self._tv_loss.set(f"{d['loss']:.4f}")
        if 'thd'    in d: self._tv_thd.set(f"{d['thd']:.2f}%")
        if 'gains'  in d:
            for o, v in self._gain_labels.items():
                if o in d['gains']:
                    kp = d['gains'][o]['kp']
                    ki = d['gains'][o]['ki']
                    v.set(f"Kp:{kp.real:.3f}{kp.imag:+.3f}j  "
                          f"Ki:{ki.real:.2f}{ki.imag:+.2f}j")
        if d.get('done'):
            # Tell _stop_tuning to leave the error text alone if it exists!
            self._stop_tuning(preserve_error="Error" in self._tv_status.get())

    # ══════════════════════════════════════════════════════════════════════════
    # Dialogs
    # ══════════════════════════════════════════════════════════════════════════
    def _save_gains(self):
        save_gains_to_file(self._tuner.gains,
                           self._transport.get_rate(),
                           self.root)

    def _open_set_gains(self):
        SetGainsDialog(self.root, self._tuner.gains,
                       self._transport.cmd_set_harmonic)

    # ══════════════════════════════════════════════════════════════════════════
    # Display update loop (15 ms)
    # ══════════════════════════════════════════════════════════════════════════
    def _schedule_update(self):
        self._update_display()
        self.root.after(15, self._schedule_update)

    def _update_display(self):
        buf = list(self._transport.sample_buf)
        ebuf = list(self._transport.error_buf)
        if not buf: return
        
        y = np.array(buf)
        n = len(y)
        yp = y[-DISPLAY_LEN:] if n >= DISPLAY_LEN else np.pad(y, (DISPLAY_LEN-n, 0))
        self._wave_line.set_data(np.arange(DISPLAY_LEN), yp)
        
        if ebuf:
            ye = np.array(ebuf)
            ne = len(ye)
            ype = ye[-DISPLAY_LEN:] if ne >= DISPLAY_LEN else np.pad(ye, (DISPLAY_LEN-ne, 0))
            self._err_line.set_data(np.arange(DISPLAY_LEN), ype)

        # FFT
        fs = self._transport.get_rate()
        nf = 1
        while nf * 2 <= min(len(y), 4096): nf *= 2
        if nf >= 64:
            seg   = y[-nf:] * np.hanning(nf)
            fft_c = np.fft.rfft(seg)
            mag   = np.abs(fft_c) * 2.0 / (nf * 0.5)
            mdb   = 20.0 * np.log10(np.maximum(mag, 1e-6))
            freqs = np.fft.rfftfreq(nf, d=1.0/fs)
            self._fft_line.set_data(freqs, mdb)

            # Clear old annotations
            for ann in self._harmonic_annotations:
                ann.remove()
            self._harmonic_annotations.clear()

            # H1 phase as reference
            f1_idx = int(round(FUND_FREQ / (fs / nf)))
            ref_phase = np.angle(fft_c[f1_idx]) if 0 < f1_idx < len(fft_c) else 0.0

            for k, vl in enumerate(self._harmonic_vlines, start=1):
                f_harm = FUND_FREQ * k
                vl.set_xdata([f_harm, f_harm])
                idx = int(round(f_harm / (fs / nf)))
                if 0 < idx < len(mdb):
                    amp_db    = mdb[idx]
                    raw_phase = np.angle(fft_c[idx])
                    rel_phase = np.degrees(raw_phase - ref_phase * k) % 360 - 180
                    ann = self._ax_fft.annotate(
                        f"H{k}\n{amp_db:.1f}dB\n{rel_phase:.1f}°",
                        xy=(f_harm, amp_db),
                        xytext=(4, 6), textcoords="offset points",
                        color="#f9e2af", fontsize=5.5,
                        fontfamily="Consolas",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1e1e2e",
                                  ec="#45475a", alpha=0.75),
                    )
                    self._harmonic_annotations.append(ann)

            self._ax_fft.set_ylim(max(float(mdb.min())-6, -100), float(mdb.max())+6)
            self._ax_fft.set_xlim(0, min(fs/2.0, 2000.0))

        # Convergence
        if self._tuner.conv_log:
            epochs = [e for e,_,_ in self._tuner.conv_log]
            losses = [l for _,l,_ in self._tuner.conv_log]
            self._conv_line.set_data(epochs, losses)
            self._ax_conv.set_xlim(0, max(epochs)+1)
            lo, hi = min(losses), max(losses)
            pad = (hi - lo) * 0.1 + 1e-9
            self._ax_conv.set_ylim(max(0, lo-pad), hi+pad)

        s = self._transport.stats
        self._sv_rate.set(f"{s['rate_hz']:>10,.0f} Hz")
        self._sv_min.set( f"{s['min_v']:>8.4f} V")
        self._sv_max.set( f"{s['max_v']:>8.4f} V")
        self._sv_mean.set(f"{s['mean_v']:>8.4f} V")
        self._sv_pkts.set(f"{s['total_packets']:>10,}")
        self._sv_drop.set(f"{s['dropped']:>10,}")

        self._canvas.draw_idle()

    def _on_close(self):
        self._tuner.running = False
        self._transport.disconnect()
        self.root.destroy()
