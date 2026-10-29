"""HeartFEM_Mu: mechanics of a heart-chamber wall from tracked images, solved by FE (legacy FEniCS) or by a
physics-informed network (PINN), with heartFEM's material and activation laws.

Four tasks (params.TASK): volume_constrain (P-V cycle of the tracked volumes), windkessel (the chamber in a closed-loop
circulation), invert_para (C and T_MAX from two tracked frames), unloading (the stress-free geometry and C from the
end-diastolic wall and a given unloaded cavity volume; FE). Each example folder holds main.py and params.py:
    python examples/<task>/<case>/main.py [steps]

Author: Siyu Mu. MIT License (LICENSE; third-party notices in THIRD_PARTY_NOTICES.md).
"""
__author__ = "Siyu Mu"
__license__ = "MIT"

from .run import run  # noqa: F401,E402
