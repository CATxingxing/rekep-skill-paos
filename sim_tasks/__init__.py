"""RoboDojo-inspired MuJoCo evaluation scenes for the Nova2 + Robotiq 2F-85 ReKep stack.

These are NOT the official RoboDojo benchmark tasks and their scores are not
comparable with the RoboDojo leaderboard.  The evaluators read simulator ground
truth; ground truth must never be given to the policy under test.  A policy only
ever sees ``TaskEnv.observe()`` (RGB, depth, joint state) and ``get_instruction()``.
"""
from .config import load_config
from .tasks import TASKS, make_env

__all__ = ["TASKS", "load_config", "make_env"]
