"""Shared Tk/ttk look of the Ortho4XP GUI: palettes, fonts and ttk styles."""

import sys
import tkinter as tk
import tkinter.font as tkfont
import tkinter.ttk as ttk

ui_theme = "light"

PALETTES = {
    "light": {
        "background": "#f3f4ef",
        "foreground": "#1f2520",
        "muted": "#5c665d",
        "border": "#c6cbc1",
        "panel_background": "#e7e9e2",
        "entry_background": "#fbfbf8",
        "entry_foreground": "#1f2520",
        "header_background": "#2c6b47",
        "header_foreground": "#f4f8f2",
        "popup_background": "#eceee7",
        "button_background": "#e2e5dc",
        "button_foreground": "#1f2520",
        "hover_background": "#d6dad0",
        "accent": "#2c6b47",
        "accent_hover": "#245a3b",
        "accent_foreground": "#f4f8f2",
        "danger": "#a8322a",
        "select_background": "#cbdfd1",
        "select_foreground": "#1f2520",
        "canvas_background": "#fbfbf8",
    },
    "dark": {
        "background": "#1d211e",
        "foreground": "#e6eae4",
        "muted": "#9aa59b",
        "border": "#3a413b",
        "panel_background": "#171a18",
        "entry_background": "#131614",
        "entry_foreground": "#e6eae4",
        "header_background": "#2f6b49",
        "header_foreground": "#eef4ec",
        "popup_background": "#252a26",
        "button_background": "#2b312c",
        "button_foreground": "#e6eae4",
        "hover_background": "#373e38",
        "accent": "#37805a",
        "accent_hover": "#408f66",
        "accent_foreground": "#f2f6f0",
        "danger": "#ef8a80",
        "select_background": "#2d4a39",
        "select_foreground": "#e6eae4",
        "canvas_background": "#131614",
    },
}

TITLE_FONT = "O4TitleFont"
SECTION_FONT = "O4SectionFont"
SMALL_FONT = "O4SmallFont"

_UI_FAMILIES = ("Segoe UI", "SF Pro Text", "Noto Sans", "DejaVu Sans")
_MONO_FAMILIES = ("Cascadia Mono", "Consolas", "SF Mono", "DejaVu Sans Mono")
_kept_fonts = []    # a tkinter Font deletes its Tk font when collected


def palette():
    """Return the active GUI palette, falling back to light for unknown values."""
    return PALETTES.get(ui_theme, PALETTES["light"])


def frame_options():
    return {"bg": palette()["background"]}


def label_options():
    colors = palette()
    return {"bg": colors["background"], "fg": colors["foreground"]}


def muted_label_options():
    colors = palette()
    return {"bg": colors["background"], "fg": colors["muted"]}


def entry_options():
    colors = palette()
    return {
        "bg": colors["entry_background"],
        "fg": colors["entry_foreground"],
        "insertbackground": colors["entry_foreground"],
        "selectbackground": colors["select_background"],
        "selectforeground": colors["select_foreground"],
        "relief": "flat",
        "highlightthickness": 1,
        "highlightbackground": colors["border"],
        "highlightcolor": colors["accent"],
    }


def text_options():
    colors = palette()
    return {
        "bg": colors["entry_background"],
        "fg": colors["foreground"],
        "insertbackground": colors["foreground"],
        "selectbackground": colors["select_background"],
        "selectforeground": colors["select_foreground"],
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
    """Heading of a group of controls."""
    colors = palette()
    return {
        "bg": colors["background"],
        "fg": colors["foreground"],
        "pady": 3,
    }


def _first_family(root, wanted):
    available = set(tkfont.families(root))
    for family in wanted:
        if family in available:
            return family
    return None


def init_fonts(root):
    """Point Tk's named fonts at the system UI face; add the heading sizes."""
    family = _first_family(root, _UI_FAMILIES)
    mono = _first_family(root, _MONO_FAMILIES)
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkCaptionFont", "TkHeadingFont"):
        font = tkfont.nametofont(name, root=root)
        if family:
            font.configure(family=family)
        font.configure(size=10)
    if mono:
        tkfont.nametofont("TkFixedFont", root=root).configure(family=mono, size=10)
    base = family or tkfont.nametofont("TkDefaultFont", root=root).actual("family")
    for name, size, weight in (
        (TITLE_FONT, 13, "bold"),
        (SECTION_FONT, 11, "bold"),
        (SMALL_FONT, 9, "normal"),
    ):
        if name not in tkfont.names(root):
            _kept_fonts.append(
                tkfont.Font(root=root, name=name, family=base, size=size, weight=weight)
            )
    root.option_add("*Font", "TkDefaultFont")


def style_window(window):
    """Give a top-level window the theme's background and title bar."""
    window.configure(bg=palette()["background"])
    if sys.platform != "win32":
        return
    try:
        import ctypes
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
        value = ctypes.c_int(1 if ui_theme == "dark" else 0)
        # DWMWA_USE_IMMERSIVE_DARK_MODE; older Windows just refuses it
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(value), ctypes.sizeof(value)
        )
    except (OSError, AttributeError, tk.TclError):
        pass


def fit_to_screen(window, width, height, min_width=640, min_height=400):
    """Size and centre a window so that all of it fits on the monitor."""
    screen_w, screen_h = window.winfo_screenwidth(), window.winfo_screenheight()
    width = min(width, screen_w - 60)
    height = min(height, screen_h - 120)   # taskbar and title bar
    x = max(0, (screen_w - width) // 2)
    y = max(0, (screen_h - height) // 2 - 30)
    window.geometry(f"{width}x{height}+{x}+{y}")
    window.minsize(min(min_width, width), min(min_height, height))


def apply_ttk_style(style=None):
    """Apply the active theme to ttk widgets."""
    colors = palette()
    style = style or ttk.Style()
    style.theme_use("clam")

    bg = colors["background"]
    border = colors["border"]
    field = colors["entry_background"]
    button = colors["button_background"]
    hover = colors["hover_background"]
    accent = colors["accent"]

    style.configure(
        ".",
        background=bg,
        foreground=colors["foreground"],
        fieldbackground=field,
        bordercolor=border,
        lightcolor=bg,
        darkcolor=bg,
        troughcolor=colors["panel_background"],
        focuscolor=border,
        selectbackground=colors["select_background"],
        selectforeground=colors["select_foreground"],
        insertcolor=colors["entry_foreground"],
        font="TkDefaultFont",
    )

    for name, padding in (("TButton", (10, 4)), ("Flat.TButton", (8, 4))):
        style.configure(
            name,
            background=button,
            foreground=colors["button_foreground"],
            bordercolor=border,
            lightcolor=button,
            darkcolor=button,
            relief="flat",
            padding=padding,
            anchor="center",
        )
        style.map(
            name,
            background=[("disabled", bg), ("pressed", hover), ("active", hover)],
            lightcolor=[("disabled", bg), ("pressed", hover), ("active", hover)],
            darkcolor=[("disabled", bg), ("pressed", hover), ("active", hover)],
            foreground=[("disabled", colors["muted"])],
            bordercolor=[("focus", accent)],
        )
    style.configure(
        "Primary.TButton",
        background=accent,
        foreground=colors["accent_foreground"],
        bordercolor=accent,
        lightcolor=accent,
        darkcolor=accent,
        relief="flat",
        padding=(14, 4),
        anchor="center",
    )
    style.map(
        "Primary.TButton",
        background=[("disabled", bg), ("pressed", colors["accent_hover"]), ("active", colors["accent_hover"])],
        lightcolor=[("disabled", bg), ("pressed", colors["accent_hover"]), ("active", colors["accent_hover"])],
        darkcolor=[("disabled", bg), ("pressed", colors["accent_hover"]), ("active", colors["accent_hover"])],
        bordercolor=[("disabled", border)],
        foreground=[("disabled", colors["muted"])],
    )
    style.configure(
        "Danger.TButton",
        background=button,
        foreground=colors["danger"],
        bordercolor=border,
        lightcolor=button,
        darkcolor=button,
        relief="flat",
        padding=(10, 4),
        anchor="center",
    )
    style.map(
        "Danger.TButton",
        background=[("pressed", hover), ("active", hover)],
        lightcolor=[("pressed", hover), ("active", hover)],
        darkcolor=[("pressed", hover), ("active", hover)],
    )
    # Small icon-like button for the "?" beside a setting
    style.configure(
        "Hint.TButton",
        background=bg,
        foreground=colors["muted"],
        bordercolor=bg,
        lightcolor=bg,
        darkcolor=bg,
        relief="flat",
        padding=(4, 0),
        width=2,
    )
    style.map(
        "Hint.TButton",
        background=[("active", hover)],
        lightcolor=[("active", hover)],
        darkcolor=[("active", hover)],
        foreground=[("active", colors["foreground"])],
    )

    # Section list beside a settings page: one looks selected, like a list row
    style.configure(
        "Nav.Toolbutton",
        background=colors["panel_background"],
        foreground=colors["foreground"],
        bordercolor=colors["panel_background"],
        lightcolor=colors["panel_background"],
        darkcolor=colors["panel_background"],
        relief="flat",
        padding=(12, 6),
        anchor="w",
    )
    style.map(
        "Nav.Toolbutton",
        background=[("selected", colors["select_background"]), ("active", hover)],
        lightcolor=[("selected", colors["select_background"]), ("active", hover)],
        darkcolor=[("selected", colors["select_background"]), ("active", hover)],
        bordercolor=[("selected", colors["select_background"]), ("active", hover)],
        foreground=[("selected", colors["select_foreground"])],
    )

    style.configure(
        "TCheckbutton",
        background=bg,
        foreground=colors["foreground"],
        indicatorbackground=field,
        indicatorforeground=colors["accent_foreground"],
        indicatorcolor=field,
        upperbordercolor=border,
        lowerbordercolor=border,
        focuscolor=bg,
        padding=(0, 2),
    )
    style.map(
        "TCheckbutton",
        background=[("active", bg)],
        indicatorcolor=[("disabled", bg), ("selected", accent)],
        indicatorbackground=[("disabled", bg), ("selected", accent)],
        foreground=[("disabled", colors["muted"])],
    )

    style.configure(
        "O4.TCombobox",
        selectbackground=field,
        selectforeground=colors["entry_foreground"],
        fieldbackground=field,
        foreground=colors["entry_foreground"],
        background=button,
        arrowcolor=colors["foreground"],
        arrowsize=13,
        bordercolor=border,
        lightcolor=field,
        darkcolor=field,
        padding=(4, 2),
    )
    style.map(
        "O4.TCombobox",
        fieldbackground=[("disabled", bg), ("readonly", field)],
        foreground=[("disabled", colors["muted"])],
        background=[("disabled", bg), ("active", hover)],
        arrowcolor=[("disabled", colors["muted"])],
        bordercolor=[("focus", accent)],
        lightcolor=[("disabled", bg)],
        darkcolor=[("disabled", bg)],
        selectbackground=[("readonly", field)],
        selectforeground=[("readonly", colors["entry_foreground"])],
    )
    style.configure(
        "TEntry",
        fieldbackground=field,
        foreground=colors["entry_foreground"],
        insertcolor=colors["entry_foreground"],
        bordercolor=border,
        lightcolor=field,
        darkcolor=field,
        padding=(4, 2),
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", bg)],
        foreground=[("disabled", colors["muted"])],
        bordercolor=[("focus", accent)],
        lightcolor=[("disabled", bg)],
        darkcolor=[("disabled", bg)],
    )
    style.configure("TLabel", background=bg, foreground=colors["foreground"])
    style.configure("TFrame", background=bg)
    style.configure("TNotebook", background=bg, bordercolor=border, tabmargins=(8, 6, 8, 0))
    style.configure(
        "TNotebook.Tab",
        background=bg,
        foreground=colors["muted"],
        bordercolor=bg,
        lightcolor=bg,
        darkcolor=bg,
        padding=(16, 6),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", colors["button_background"]), ("active", hover)],
        lightcolor=[("selected", colors["button_background"]), ("active", hover)],
        foreground=[("selected", colors["foreground"]), ("active", colors["foreground"])],
        bordercolor=[("selected", border)],
    )
    for name in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(
            name,
            background=button,
            troughcolor=bg,
            bordercolor=bg,
            lightcolor=button,
            darkcolor=button,
            arrowcolor=colors["muted"],
            relief="flat",
        )
        style.map(
            name,
            background=[("active", hover), ("disabled", bg)],
            lightcolor=[("active", hover), ("disabled", bg)],
            darkcolor=[("active", hover), ("disabled", bg)],
        )
    style.configure(
        "Treeview",
        background=field,
        fieldbackground=field,
        foreground=colors["foreground"],
        bordercolor=border,
    )
    style.configure(
        "Treeview.Heading",
        background=button,
        foreground=colors["button_foreground"],
        bordercolor=border,
        lightcolor=button,
        darkcolor=button,
        relief="flat",
        padding=(6, 3),
    )
    style.map("Treeview.Heading", background=[("active", hover)])

    # The list a combobox drops down is a classic Tk listbox
    root = style.master
    root.option_add("*TCombobox*Listbox.background", field)
    root.option_add("*TCombobox*Listbox.foreground", colors["entry_foreground"])
    root.option_add("*TCombobox*Listbox.selectBackground", colors["select_background"])
    root.option_add("*TCombobox*Listbox.selectForeground", colors["select_foreground"])
    root.option_add("*TCombobox*Listbox.relief", "flat")
    root.option_add("*TCombobox*Listbox.font", "TkDefaultFont")
    return style
