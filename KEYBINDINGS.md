# Keyboard shortcut

Bind a key to `sniptext --capture-now` in your compositor or desktop environment.

**Hyprland** (`~/.config/hypr/hyprland.conf`):

```
bind = CTRL ALT, T, exec, sniptext --capture-now
```

**Sway** (`~/.config/sway/config`) and **i3** (`~/.config/i3/config`):

```
bindsym Ctrl+Alt+t exec sniptext --capture-now
```

**GNOME**: Settings → Keyboard → Keyboard Shortcuts → Custom Shortcuts.

**KDE Plasma**: System Settings → Shortcuts → Custom Shortcuts → Command/URL.
