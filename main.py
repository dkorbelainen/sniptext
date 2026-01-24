"""
SnipText - Wrapper for backward compatibility.
Use 'python -m sniptext' or 'sniptext' command instead.
"""

import sys
from sniptext.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
