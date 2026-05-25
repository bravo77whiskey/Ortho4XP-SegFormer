"""Shared Tk/ttk color theme helpers for the Ortho4XP GUI."""

import tkinter.ttk as ttk

ui_theme = "light"

PALETTES = {
    "light": {
        "background": "light green",
        "foreground": "black",
        "entry_background": "white",
        "entry_foreground": "blue",
        "header_background": "dark green",
        "header_foreground": "light green",
        "popup_background": "light gray",
        "button_background": "light green",
        "button_foreground": "black",
        "select_background": "white",
        "select_foreground": "blue",
        "canvas_background": "white",
    },
    "dark": {
        "background": "#1f2420",
        "foreground": "#f1f5ef",
        "entry_background": "#111611",
        "entry_foreground": "#b8d7ff",
        "header_background": "#0d351f",
        "header_foreground": "#d6f5d4",
        "popup_background": "#242a24",
        "button_background": "#2f3a31",
        "button_foreground": "#f1f5ef",
        "select_background": "#233426",
        "select_foreground": "#f1f5ef",
        "canvas_background": "#111611",
    },
}


def palette():
    """Return the active GUI palette, falling back to light for unknown values."""
    return PALETTES.get(ui_theme, PALETTES["light"])


def frame_options():
    return {"bg": palette()["background"]}


def label_options():
    colors = palette()
    return {"bg": colors["background"], "fg": colors["foreground"]}


def entry_options():
    colors = palette()
    return {
        "bg": colors["entry_background"],
        "fg": colors["entry_foreground"],
        "insertbackground": colors["entry_foreground"],
    }


def checkbutton_options():
    colors = palette()
    return {
        "bg": colors["background"],
        "fg": colors["foreground"],
        "activebackground": colors["background"],
        "activeforeground": colors["foreground"],
        "selectcolor": colors["entry_background"],
        "highlightthickness": 0,
    }


def header_options():
    colors = palette()
    return {
        "bg": colors["header_background"],
        "fg": colors["header_foreground"],
    }


def apply_ttk_style(style=None):
    """Apply the active theme to ttk widgets used by the legacy Tk GUI."""
    colors = palette()
    style = style or ttk.Style()
    style.theme_use("alt")
    style.configure(
        ".",
        background=colors["background"],
        foreground=colors["foreground"],
        fieldbackground=colors["entry_background"],
    )
    style.configure(
        "TButton",
        background=colors["button_background"],
        foreground=colors["button_foreground"],
        focuscolor=colors["background"],
    )
    style.map(
        "TButton",
        background=[
            ("active", colors["select_background"]),
            ("disabled", colors["background"]),
        ],
        foreground=[("disabled", colors["foreground"])],
    )
    style.configure(
        "Flat.TButton",
        background=colors["button_background"],
        foreground=colors["button_foreground"],
        highlightbackground=colors["background"],
        selectbackground=colors["background"],
        highlightcolor=colors["background"],
        highlightthickness=0,
        relief="flat",
    )
    style.map(
        "Flat.TButton",
        background=[
            ("active", colors["select_background"]),
            ("disabled", colors["background"]),
        ],
        foreground=[("disabled", colors["foreground"])],
    )
    style.configure(
        "O4.TCombobox",
        selectbackground=colors["select_background"],
        selectforeground=colors["select_foreground"],
        fieldbackground=colors["entry_background"],
        foreground=colors["entry_foreground"],
        background=colors["entry_background"],
        arrowcolor=colors["foreground"],
    )
    style.map(
        "O4.TCombobox",
        fieldbackground=[
            ("readonly", colors["entry_background"]),
            ("disabled", colors["background"]),
            ("active", colors["entry_background"]),
        ],
        foreground=[("disabled", colors["foreground"])],
        selectbackground=[("readonly", colors["select_background"])],
        selectforeground=[("readonly", colors["select_foreground"])],
    )
    style.configure(
        "TEntry",
        fieldbackground=colors["entry_background"],
        foreground=colors["entry_foreground"],
        insertcolor=colors["entry_foreground"],
    )
    style.configure(
        "TLabel",
        background=colors["background"],
        foreground=colors["foreground"],
    )
    style.configure("TFrame", background=colors["background"])
    style.configure("TNotebook", background=colors["background"])
    style.configure(
        "TNotebook.Tab",
        background=colors["button_background"],
        foreground=colors["button_foreground"],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors["background"])],
        foreground=[("selected", colors["foreground"])],
    )
    return style
