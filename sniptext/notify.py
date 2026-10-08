"""Desktop notifications through notify-send."""

import subprocess

from loguru import logger

_PREVIEW_LEN = 50


def preview(text: str) -> str:
    """The start of a text on one line, for a notification body."""
    head = text[:_PREVIEW_LEN].replace("\n", " ")
    return head + ("…" if len(text) > _PREVIEW_LEN else "")


def send(message: str) -> None:
    """Show a notification; a missing or failing notify-send is not an error."""
    try:
        result = subprocess.run(
            ["notify-send", "SnipText", message], timeout=2, capture_output=True, check=False
        )
        if result.returncode != 0:
            logger.debug(f"notify-send failed with code {result.returncode}")
    except FileNotFoundError:
        logger.debug("notify-send not found - install libnotify package")
    except Exception as e:
        logger.debug(f"Could not show notification: {e}")
