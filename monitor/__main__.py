"""
__main__.py — Entry point.
Run with:  python -m monitor
"""

import tkinter as tk
from tkinter import ttk
from .app import MonitorApp


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground="#313244", background="#45475a",
                    foreground="#cdd6f4", selectbackground="#45475a",
                    selectforeground="#cdd6f4", arrowcolor="#cdd6f4")
    style.map("TCombobox", fieldbackground=[("readonly", "#313244")])
    MonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
