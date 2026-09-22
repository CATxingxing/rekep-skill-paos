"""Lightweight shared exceptions (importable without pulling in MuJoCo/Torch)."""


class UnreachablePose(RuntimeError):
    """A commanded end-effector pose has no IK solution for this robot."""
