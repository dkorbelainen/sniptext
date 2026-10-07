"""The candidate pipelines the router's actions are selected from."""

from __future__ import annotations

from sniptext.pipelines import Pipeline

_TRANSFORMS = {
    "enhance": ("enhance",),
    "light": ("light",),
    "light_up2": ("light", "up2"),
    "light_up2_median3": ("light", "up2", "median3"),
    "light_gauss1": ("light", "gauss1"),
    "light_median3": ("light", "median3"),
}


def _name(transform: str, psm: str) -> str:
    return f"{transform}_auto" if psm == "auto" else f"{transform}_psm{psm}"


POOL: tuple[Pipeline, ...] = (
    *(
        Pipeline(_name(transform, psm), steps, psm)
        for transform, steps in _TRANSFORMS.items()
        for psm in ("auto", "6")
    ),
    Pipeline("light_psm11", ("light",), "11"),
)
