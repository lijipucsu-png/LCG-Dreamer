"""Capture real SUMO frames and compose the publication storyboards."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from final_model.sumo_lane_change_storyboard import main


if __name__ == "__main__":
    main()
