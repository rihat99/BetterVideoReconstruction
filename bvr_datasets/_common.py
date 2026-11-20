"""What the force scorers of several datasets share: the fold of body joints onto the four limbs."""

from __future__ import annotations

_FINGER = ("index", "middle", "pinky", "ring", "thumb")


def _limb_members(joint_names: list[str]) -> list[list[int]]:
    """Joint columns per limb, in the order left hand, right hand, left foot, right foot.

    A hand is the wrist and its fingers, a foot the foot and the ankle, by name. A name the list does not hold is
    skipped, so the same rule folds the body's 52 joints and a shorter list of the joints a solve loads.
    """
    index = {name: column for column, name in enumerate(joint_names)}
    present = lambda *names: [index[name] for name in names if name in index]  # noqa: E731
    hands = [present(f"{side}_wrist")
             + [column for name, column in index.items()
                if name.startswith(f"{side}_") and any(finger in name for finger in _FINGER)]
             for side in ("left", "right")]
    feet = [present(f"{side}_foot", f"{side}_ankle") for side in ("left", "right")]
    return hands + feet
