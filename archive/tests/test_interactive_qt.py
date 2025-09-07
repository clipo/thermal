#!/usr/bin/env python3
"""
Interactive SGD viewer using Qt backend as fallback.
"""

import matplotlib
try:
    matplotlib.use('Qt5Agg')  # Try Qt5 backend
except:
    try:
        matplotlib.use('QtAgg')  # Try Qt backend
    except:
        matplotlib.use('TkAgg')  # Fall back to Tk

import matplotlib.pyplot as plt
from test_interactive_sgd import test_interactive

if __name__ == "__main__":
    print(f"Using matplotlib backend: {matplotlib.get_backend()}")
    test_interactive()