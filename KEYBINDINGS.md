# Keyboard Shortcut Setup

SnipText requires you to configure a keyboard shortcut in your window manager or desktop environment.

**Recommended keybind: `Ctrl+Alt+T`**

Configure your system to run this command when the keybind is pressed:
```bash
sniptext --capture-now
```

## Configuration Examples

### Wayland Compositors

**Hyprland** (`~/.config/hypr/hyprland.conf`):
```
bind = CTRL ALT, T, exec, sniptext --capture-now
```

**Sway** (`~/.config/sway/config`):
```
bindsym Ctrl+Alt+t exec sniptext --capture-now
```

### X11 Window Managers

**i3** (`~/.config/i3/config`):
```
bindsym Ctrl+Alt+t exec sniptext --capture-now
```

### Desktop Environments

**GNOME**: Settings → Keyboard → Keyboard Shortcuts → Custom Shortcuts
- Click "+" to add new shortcut
- Name: `SnipText OCR`
- Command: `sniptext --capture-now`
- Set shortcut: Press `Ctrl+Alt+T`

**KDE Plasma**: System Settings → Shortcuts → Custom Shortcuts
- Right-click → New → Global Shortcut → Command/URL
- Name: `SnipText OCR`
- Trigger: Set to `Ctrl+Alt+T`
- Action: `sniptext --capture-now`

Feel free to use any keybind you prefer (e.g., `Super+Shift+S`).

## Why Manual Setup?

On Wayland, applications cannot register global hotkeys due to security restrictions. Even on X11, WM-level shortcuts are more reliable than application-level hotkeys.
