"""Capture continuous-lateral SUMO lane-change publication figures."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final_model.sumo_continuous_lane_change_storyboard import main


if __name__ == "__main__":
    main()
