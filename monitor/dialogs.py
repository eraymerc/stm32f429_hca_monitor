"""
dialogs.py — Modal dialogs: manual gain editor and save-gains-to-file.
"""

import cmath
import math
import datetime
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from .config import FUND_FREQ, SAMPLE_RATE_NOM


def save_gains_to_file(gains: dict, sample_rate: float, parent: tk.Tk):
    """Write current harmonic gains to a user-chosen .txt file."""
    if not gains:
        return
    filename = filedialog.asksaveasfilename(
        title="Save Harmonic Gains", parent=parent,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not filename:
        return
    try:
        with open(filename, 'w') as f:
            f.write("Harmonic Controller Gains\n")
            f.write(f"Generated : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Sample rate: {sample_rate:.2f} Hz\n")
            f.write(f"Fund. freq : {FUND_FREQ} Hz\n\n")
            hdr = (f"{'Order':<8}{'Kp Real':>12}{'Kp Imag':>12}"
                   f"{'|Kp|':>12}{'Kp Ang°':>12}"
                   f"{'Ki Real':>12}{'Ki Imag':>12}"
                   f"{'|Ki|':>12}{'Ki Ang°':>12}")
            f.write(hdr + "\n" + "-" * len(hdr) + "\n")
            for order in sorted(gains.keys()):
                kp, ki = gains[order]['kp'], gains[order]['ki']
                km_p, ka_p = cmath.polar(kp)
                km_i, ka_i = cmath.polar(ki)
                f.write(
                    f"{order:<8}"
                    f"{kp.real:>12.6f}{kp.imag:>12.6f}"
                    f"{km_p:>12.6f}{math.degrees(ka_p):>12.2f}"
                    f"{ki.real:>12.6f}{ki.imag:>12.6f}"
                    f"{km_i:>12.6f}{math.degrees(ka_i):>12.2f}\n"
                )
        return True
    except Exception as e:
        messagebox.showerror("Save Error", str(e), parent=parent)
        return False


class SetGainsDialog:
    """
    Modal window for manually editing Kp / Ki of any harmonic order.

    Usage:
        dlg = SetGainsDialog(parent, gains, send_fn)
        # blocks until window is closed
    """

    def __init__(self, parent: tk.Tk, gains: dict, send_fn):
        """
        gains   — the shared gain dict {order: {'kp': complex, 'ki': complex}}
        send_fn — callable(order, kp, ki) -> bool
        """
        self._gains   = gains
        self._send_fn = send_fn

        win = tk.Toplevel(parent)
        win.title("Set Harmonic Gains")
        win.configure(bg="#1e1e2e")
        win.resizable(False, False)

        L = dict(bg="#1e1e2e", fg="#cdd6f4", font=("Segoe UI", 9))
        E = dict(bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4",
                 relief=tk.FLAT, font=("Consolas", 9), width=12)

        # Order selector
        f1 = tk.Frame(win, bg="#1e1e2e", pady=8, padx=10)
        f1.pack(fill=tk.X)
        tk.Label(f1, text="Harmonic Order:", **L).pack(side=tk.LEFT, padx=(0, 6))
        order_list = sorted(gains.keys()) if gains else [0]
        self._order_var = tk.StringVar(value=str(order_list[0]))
        self._order_cb = ttk.Combobox(f1, textvariable=self._order_var,
                                      values=[str(o) for o in order_list], width=10)
        self._order_cb.pack(side=tk.LEFT)

        # Gain entries (Magnitude and Phase)
        f2 = tk.Frame(win, bg="#1e1e2e", padx=10, pady=4)
        f2.pack(fill=tk.X)
        self._entries: dict[str, tk.Entry] = {}
        for text, key, col in [("Kp Mag","kp_m",0),("Kp Phase°","kp_p",1),
                                ("Ki Mag","ki_m",2),("Ki Phase°","ki_p",3)]:
            tk.Label(f2, text=text+":", **L).grid(
                row=0, column=col*2, padx=4, pady=4, sticky=tk.E)
            ent = tk.Entry(f2, **E)
            ent.grid(row=0, column=col*2+1, padx=4, pady=4)
            self._entries[key] = ent

        self._order_var.trace_add('write', lambda *_: self._populate())
        self._populate()

        # Buttons
        bf = tk.Frame(win, bg="#1e1e2e", pady=10)
        bf.pack()
        tk.Button(bf, text="Apply", bg="#a6e3a1", fg="#1e1e2e", relief=tk.FLAT,
                  font=("Segoe UI", 9, "bold"), cursor="hand2",
                  command=self._apply).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Close", bg="#45475a", fg="#cdd6f4", relief=tk.FLAT,
                  font=("Segoe UI", 9), cursor="hand2",
                  command=win.destroy).pack(side=tk.LEFT, padx=4)

        # Center on parent and make modal
        win.transient(parent); win.grab_set()
        win.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width()  - win.winfo_width())  // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")
        self._win = win

    def _populate(self):
        try: order = int(self._order_var.get())
        except ValueError: return
        kp = self._gains.get(order, {}).get('kp', complex(0.0))
        ki = self._gains.get(order, {}).get('ki', complex(0.0))
        
        km_p, ka_p = cmath.polar(kp)
        km_i, ka_i = cmath.polar(ki)

        for key, val in [('kp_m', km_p), ('kp_p', math.degrees(ka_p)),
                         ('ki_m', km_i), ('ki_p', math.degrees(ka_i))]:
            e = self._entries[key]
            e.delete(0, tk.END); e.insert(0, f"{val:.6f}")

    def _apply(self):
        try:
            order = int(self._order_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid order.", parent=self._win)
            return
        try:
            kp_m = float(self._entries['kp_m'].get())
            kp_p = math.radians(float(self._entries['kp_p'].get()))
            kp = cmath.rect(kp_m, kp_p)

            ki_m = float(self._entries['ki_m'].get())
            ki_p = math.radians(float(self._entries['ki_p'].get()))
            ki = cmath.rect(ki_m, ki_p)
        except ValueError:
            messagebox.showerror("Error", "All fields must be floats.",
                                 parent=self._win)
            return

        if order not in self._gains:
            self._gains[order] = {}
        self._gains[order]['kp'] = kp
        self._gains[order]['ki'] = ki

        ok = self._send_fn(order, kp, ki)
        if ok:
            messagebox.showinfo("Success",
                                f"Gains for H{order} sent to MCU.",
                                parent=self._win)
        else:
            messagebox.showwarning("Not Connected",
                                   "Gains saved locally — not sent "
                                   "(no active connection).",
                                   parent=self._win)

        # Refresh order list
        vals = sorted(set(list(self._gains.keys())))
        self._order_cb['values'] = [str(v) for v in vals]
        self._order_var.set(str(order))