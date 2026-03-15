import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
runpy.run_path(str(Path(__file__).parent.parent / "src/sneaker_intel/dashboard/pages/01_market_overview.py"))
