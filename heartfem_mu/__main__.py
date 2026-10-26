"""One pipeline step in its own process, started by run.py:  python -m heartfem_mu <example folder> <step> [args]"""
import sys

from .run import run_step

run_step(sys.argv[1], sys.argv[2], sys.argv[3:])
