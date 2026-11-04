"""Run this example (settings in params.py):  python main.py [steps | --from <step> | --list]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # HeartFEM_Mu/
import heartfem_mu  # noqa: E402

heartfem_mu.run(__file__, sys.argv[1:])
