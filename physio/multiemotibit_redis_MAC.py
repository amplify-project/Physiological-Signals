"""Compatibility launcher for the current EmotiBit SD/RF pipeline.

The project-level physiological pipeline now lives in
``multiemotibit_UDP_SD_RFv2.py``. This module keeps the previous entry point
working for existing docs, scripts, and habits.
"""

from multiemotibit_UDP_SD_RFv2 import main


if __name__ == "__main__":
    main()
