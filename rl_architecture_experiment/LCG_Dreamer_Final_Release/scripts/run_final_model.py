"""Run the final release without requiring an editable installation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from final_model.demo import main


if __name__ == "__main__":
    main()
