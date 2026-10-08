"""Write docs/benchmark.md and its figure from the benchmark outputs.

Every number in the report comes from router_eval.json or corrector_eval.json;
nothing is typed in by hand.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sniptext.pipelines import LARGE_IMAGE, V04

_HERE = Path(__file__).resolve().parent
_EVAL = _HERE / "router_eval.json"
_CORRECTOR = _HERE / "corrector_eval.json"
_ENVIRONMENT = _HERE / "data" / "browser" / "environment.json"
_REPO_ROOT = _HERE.parent
_REPORT = _REPO_ROOT / "docs" / "benchmark.md"
_IMG = _REPO_ROOT / "docs" / "img"

_FIXED = {
    "router_pre_ocr": "Router, before OCR",
    "router_cascade": "Router, cascade",
    "router_v04": "0.4 router (needs EasyOCR)",
    "oracle": "Oracle over the shipped pipelines",
    "oracle_pool": "Oracle over the whole pool",
}
_SLICES = (
    ("test", "Result on held-out texts", ""),
    ("unseen_font", "Unseen fonts", "Two font families that appear in no other slice.\n"),
    (
        "ood",
        "Out-of-domain receipts",
        "Photographed receipts are unlike anything in the training data. This slice shows what "
        "the choice does outside its domain.\n",
    ),
    (
        "browser",
        "Pages rendered by a browser",
        "Held-out texts laid out by headless Chrome as prose, highlighted code and interface "
        "elements, in light and dark themes, at device scale 1, 1.5 and 2, with no degradation "
        "added. Nothing was fitted or selected on this slice.\n",
    ),
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _ci(triple) -> str:
    mean, low, high = triple
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def _signed(triple) -> str:
    mean, low, high = triple
    return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"


def _spec(spec: dict) -> str:
    if spec["kind"] == "ridge":
        return f"Ridge regression (alpha {spec['alpha']:g})"
    return (
        f"Gradient boosting ({spec['n_estimators']} trees, depth {spec['max_depth']}, "
        f"learning rate {spec['learning_rate']:g})"
    )


def _shipped_label(ev: dict) -> str:
    if ev["shipped"] == "static":
        return f"always_{ev['best_static']}"
    return f"router_{ev['shipped']}"


def _label(policy: str, ev: dict) -> str:
    if policy in _FIXED:
        return _FIXED[policy]
    name = policy[len("always_") :]
    notes = [
        note
        for note, applies in (
            ("0.4 Tesseract", name == V04.name),
            ("best static", name == ev["best_static"]),
        )
        if applies
    ]
    return f"Always `{name}`" + (f" ({', '.join(notes)})" if notes else "")


def verdict(summary: dict, ev: dict) -> str:
    """One sentence that says only what the paired interval supports."""
    name = _label(summary["policy"], ev)
    if "delta" not in summary:
        return (
            f"No router ships. {name}: CER {_ci(summary['cer'])}; it is the best static "
            "pipeline, so there is nothing to compare it with."
        )
    mean, low, high = summary["delta"]
    static = f"always running `{ev['best_static']}`"
    if high < 0:
        relation = f"lower than {static} by {-mean:.3f}"
    elif low > 0:
        relation = f"higher than {static} by {mean:.3f}"
    else:
        relation = f"not distinguishable from {static}"
    return (
        f"{name}: CER {_ci(summary['cer'])}, {relation} "
        f"(paired difference {_signed(summary['delta'])}, 95% interval over texts)."
    )


def _slice_table(summaries: list, ev: dict) -> str:
    lines = [
        "| Policy | CER, clipped at 1 | CER, raw | Median | Difference to best static "
        "| Pipelines taken | Time, ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        label = _label(s["policy"], ev)
        if s["policy"] == _shipped_label(ev):
            label = f"**{label} (shipped)**"
        share = ", ".join(
            f"{value * 100:.0f}% `{name}`" for name, value in s.get("share", {}).items() if value
        )
        delta = _signed(s["delta"]) if "delta" in s else ""
        lines.append(
            f"| {label} | {_ci(s['cer'])} | {s['cer_unclipped']:.3f} | {s['cer_median']:.3f} "
            f"| {delta} | {share} | {s['time'] * 1000:.0f} |"
        )
    return "\n".join(lines) + "\n"


def _slice_section(ev: dict, key: str) -> list:
    """Verdict and table of one slice; nothing when the run has no such slice."""
    if key not in ev["slices"]:
        return []
    summaries = ev["slices"][key]
    shipped = next(s for s in summaries if s["policy"] == _shipped_label(ev))
    out = [verdict(shipped, ev) + "\n", _slice_table(summaries, ev)]
    if key in ev["per_slice_delta_v04_tesseract"]:
        out.append(
            f"Against Tesseract as version 0.4 ran it (`{V04.name}`) the paired difference is "
            f"{_signed(ev['per_slice_delta_v04_tesseract'][key])}.\n"
        )
    if key in ev["per_slice_delta_v04"]:
        out.append(
            "Against the 0.4 router the paired difference is "
            f"{_signed(ev['per_slice_delta_v04'][key])}. Its times come from the 0.4 run, with "
            "EasyOCR on a GPU, and are not comparable with the other rows.\n"
        )
    return out


def _criteria(ev: dict) -> str:
    c = ev["criteria"]

    def line(text: str, met: bool, delta) -> str:
        numbers = f" (paired difference {_signed(delta)})" if delta else ""
        return f"{text}: {'**met**' if met else '**not met**'}{numbers}."

    return "\n".join(
        [
            line("1. The shipped router has significantly lower CER on held-out texts than the "
                 "best static pipeline", c["router_beats_best_static"], c["test_delta"]),
            line("2. It is not significantly worse than the 0.4 router, which needs EasyOCR",
                 c["not_worse_than_v04"], c["v04_delta"]),
            line("3. On browser pages it is not significantly worse than the best static "
                 "pipeline", c["browser_not_worse"], c["browser_delta"]),
        ]
    ) + "\n"  # fmt: skip


def _group_table(groups: dict, ev: dict) -> str:
    policies = [k for k in next(iter(groups.values())) if k != "n"]
    lines = [
        "| Group | n | " + " | ".join(_label(p, ev) for p in policies) + " |",
        "|---|---|" + "---|" * len(policies),
    ]
    for group, values in groups.items():
        cells = " | ".join(f"{values[p]:.3f}" for p in policies)
        lines.append(f"| {group} | {values['n']} | {cells} |")
    return "\n".join(lines) + "\n"


def _pool_table(ev: dict) -> str:
    lines = [
        "| Pipeline | Steps | PSM | CER on train and validation | Time, ms |",
        "|---|---|---|---|---|",
    ]
    for pipeline in sorted(ev["pool"], key=lambda p: ev["pool_dev_cer"][p["name"]]):
        name = pipeline["name"]
        mark = " (selected)" if name in ev["actions"] else ""
        lines.append(
            f"| `{name}`{mark} | {', '.join(pipeline['steps'])} | {pipeline['psm']} "
            f"| {ev['pool_dev_cer'][name]:.3f} | {ev['pool_seconds'][name] * 1000:.0f} |"
        )
    return "\n".join(lines) + "\n"


def _steps_table(ev: dict) -> str:
    lines = ["| Step | Pipeline | Out-of-fold CER of the routed set | Oracle over the set |",
             "|---|---|---|---|"]  # fmt: skip
    for number, step in enumerate(ev["action_selection"], 1):
        name = f"`{step['added']}`" if step["added"] else f"`{step['rejected']}` (rejected)"
        lines.append(f"| {number} | {name} | {step['oof_cer']:.3f} | {step['oracle']:.3f} |")
    return "\n".join(lines) + "\n"


def _selection_table(selection: list) -> str:
    lines = ["| Candidate | Out-of-fold CER | Regret to oracle |", "|---|---|---|"]
    for entry in sorted(selection, key=lambda e: e["oof_regret"]):
        lines.append(
            f"| {_spec(entry['spec'])} | {entry['oof_cer']:.3f} | {entry['oof_regret']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _feature_table(policy: dict) -> str:
    lines = [
        "| Input | CER change when removed (out-of-fold) | CER change when shuffled (test) |",
        "|---|---|---|",
    ]
    for name in sorted(policy["feature_names"], key=lambda n: -policy["ablation"][n]):
        lines.append(
            f"| {name} | {policy['ablation'][name]:+.3f} | {policy['importance'][name]:+.3f} |"
        )
    return "\n".join(lines) + "\n"


# Chart tokens: a categorical pair (blue, orange) checked for colour-vision separation
# and contrast on this light surface; text uses ink tokens, never a series colour.


_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e6e5e1"
_SERIES = ("#2a78d6", "#eb6834")


def _new_axes(figsize):
    """A figure on the chart surface with a hairline grid and recessive axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_SECONDARY, length=0, labelsize=9)
    return plt, fig, ax


def _finish(plt, fig, ax, path: Path, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, color=_INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=_INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, labelcolor=_INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _plot_curve(ev: dict, path: Path) -> None:
    """Mean test CER against mean time per image, one line per policy over the time weight."""
    plt, fig, ax = _new_axes((7, 4.2))
    for (policy, values), color in zip(ev["policies"].items(), _SERIES):
        curve = sorted(values["curve"], key=lambda point: point[2])
        ax.plot(
            [p[2] * 1000 for p in curve],
            [p[1] for p in curve],
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.5,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=_FIXED[f"router_{policy}"],
        )
    for summary in ev["slices"]["test"]:
        policy = summary["policy"]
        if not (policy.startswith("always_") or policy == "oracle"):
            continue
        point = (summary["time"] * 1000, summary["cer"][0])
        ax.plot(*point, linestyle="none", marker="D", markersize=6, color=_INK_SECONDARY,
                markeredgecolor=_SURFACE, markeredgewidth=1.5)  # fmt: skip
        # The oracle sits under the router curves; its label goes below the marker.
        below = policy == "oracle"
        label = "Oracle" if below else policy[len("always_") :]
        ax.annotate(label, point, textcoords="offset points", xytext=(6, -6 if below else 6),
                    va="top" if below else "baseline", fontsize=8, color=_INK)  # fmt: skip
    ax.set_ylim(bottom=0)
    _finish(plt, fig, ax, path, "Mean time per image, ms", "Mean CER on held-out texts")


def render(ev: dict, corrector: dict, environment: dict | None, commit: str, generated: str) -> str:
    card = ev["dataset"]
    routed = ev["shipped"] != "static"
    splits = ", ".join(f"{name} {count}" for name, count in card["by_split"].items())
    actions = ", ".join(f"`{name}`" for name in ev["actions"])
    load_start, load_end = (f"{values[0]:.1f}" for values in ev["loadavg"])

    out = [
        "# Tesseract pipeline router benchmark\n",
        f"Generated: {generated}. Commit: `{commit}`. Tesseract runs with `{card['language']}`.\n",
        "## Task\n",
        "SnipText reads every capture with Tesseract. What varies is the pipeline: how the image "
        "is prepared (inversion of dark themes, upscaling, median or Gaussian filtering, or the "
        "preprocessing of version 0.4) and which page-segmentation mode Tesseract gets. For each "
        "image the router predicts the character error rate (CER) of each shipped pipeline and "
        "takes the one minimising predicted CER plus a time weight times the pipeline's seconds. "
        "Two routers are compared: one that sees only image statistics before any OCR, and a "
        "cascade that runs the default pipeline first and also sees its word confidences.\n",
        "CER is the edit distance to the ground truth over its length, whitespace-normalised, "
        "case kept. Averages use CER clipped at 1 so that one garbage output does not dominate; "
        "the raw mean is shown next to it. Intervals are 95% bootstrap intervals that resample "
        "texts, not images, because renders of one text are not independent.\n",
        "If the chosen pipeline returns no text the default pipeline runs instead; the numbers "
        "below include that rule and its time.\n",
        "## Data\n",
        f"{card['n']} images: {splits}. {card['texts']} distinct texts "
        f"({card['by_lang']['en']} English and {card['by_lang']['ru']} Russian renders; "
        f"prose {card['by_content']['prose']}, code {card['by_content']['code']}, "
        f"interface strings {card['by_content']['ui']}). Sentences come from UD English-EWT and "
        "UD Russian-GSD, code from the CPython 3.12 standard library. Each text is rendered "
        "several times with a random font, size from 11 to 30 px, one of six colour schemes and "
        "up to two degradations out of blur, noise, JPEG, rescaling and low contrast.\n",
        "A text belongs to exactly one of train, validation and test. Two font families appear "
        "only in the unseen-font slice. The receipts (SROIE) and the browser pages are never "
        "used for fitting or selection.\n",
    ]
    if environment:
        fonts = "; ".join(f"{family}: {match}" for family, match in environment["fonts"].items())
        out.append(f"The browser pages were rendered by {environment['chrome']}. Fonts: {fonts}.\n")

    for key, title, intro in _SLICES:
        section = _slice_section(ev, key)
        if not section:
            continue
        out.append(f"## {title}\n")
        if intro:
            out.append(intro)
        out += section
        if key == "test":
            if routed:
                ship = ev["policies"][ev["shipped"]]
                other = "cascade" if ev["shipped"] == "pre_ocr" else "pre_ocr"
                mine, theirs = ship["oof"], ev["policies"][other]["oof"]
                out += [
                    f"The shipped policy is `{ev['shipped']}` over {actions}: "
                    f"{_spec(ship['spec'])}, time weight {ship['time_weight']:.3f}. It is scored "
                    "through `sniptext.router.Router`, the class the app uses, reading the model "
                    "file packaged with the app.\n",
                    "Of the two routers the one with the lower out-of-fold CER ships, or the "
                    f"faster one when they are within 0.005: {_FIXED['router_' + ev['shipped']]} "
                    f"has {mine['cer']:.3f} at {mine['time'] * 1000:.0f} ms per image, "
                    f"{_FIXED['router_' + other]} {theirs['cer']:.3f} at "
                    f"{theirs['time'] * 1000:.0f} ms.\n",
                ]
            out += ["The criteria fixed before the measurement:\n", _criteria(ev)]

    out += [
        "## Clean and degraded images\n",
        "Mean clipped CER on held-out texts, split by whether a degradation was applied. The "
        "router is credited only where it changes something.\n",
        _group_table(ev["clean_vs_degraded"], ev),
    ]

    if routed:
        spans = []
        for policy, values in ev["policies"].items():
            times = [point[2] * 1000 for point in values["curve"]]
            cers = [point[1] for point in values["curve"]]
            spans.append(
                f"{_FIXED['router_' + policy]} spans {min(times):.0f} to {max(times):.0f} ms and "
                f"CER {min(cers):.3f} to {max(cers):.3f}"
            )
        out += [
            "## Accuracy against time\n",
            "![CER against time](img/cer_time.png)\n",
            "Each line traces one router as the time weight grows from 0. "
            + "; ".join(spans)
            + ". The cascade runs the default pipeline on every image, so its time cannot fall "
            "below that pipeline's.\n",
        ]
    out.append(
        "Times were measured in one process on one machine; its load average was "
        f"{load_start} when the timing pass started and {load_end} when it ended. Feature "
        f"extraction takes {ev['feature_time_mean'] * 1000:.1f} ms per image and is not included "
        "in the router's time.\n"
    )

    out += [
        "## How the pipelines were chosen\n",
        f"The candidate pool has {len(ev['pool'])} pipelines. `{V04.name}` is what version 0.4 "
        "ran.\n",
        _pool_table(ev),
        "Selection is greedy on train and validation texts: start from the best static "
        "pipeline, then add the candidate that lowers the out-of-fold CER of the routed set "
        "most, and stop when the gain is below 0.003 or at four pipelines.\n",
        _steps_table(ev),
    ]

    if routed:
        out += [
            "## Model selection\n",
            "Candidates are compared by grouped 5-fold cross-validation over train and "
            "validation texts (a text is never in both the fitting and the predicted fold), on "
            "the regret of the resulting policy to the oracle. The default time weight is the "
            "largest one that keeps out-of-fold CER within 0.005 of the accuracy-only policy.\n",
        ]
        for policy, values in ev["policies"].items():
            out += [
                f"### {_FIXED['router_' + policy]}\n",
                _selection_table(values["selection"]),
                f"Out-of-fold at its default time weight {values['time_weight']:.3f}: CER "
                f"{values['oof']['cer']:.3f}, {values['oof']['time'] * 1000:.0f} ms per image.\n",
            ]
        out += [
            "## Features\n",
            "Inputs of the shipped router.\n",
            _feature_table(ev["policies"][ev["shipped"]]),
        ]

    out += ["## Breakdown\n", "Mean clipped CER per group.\n"]
    titles = {"degradation": "By degradation", "theme": "By colour scheme", "lang": "By language",
              "content": "By content", "scale": "By device scale"}  # fmt: skip
    for slice_name, heading in (("test", "Held-out texts"), ("browser", "Browser pages")):
        for key, groups in ev["breakdown"].get(slice_name, {}).items():
            out += [f"### {heading}: {titles[key].lower()}\n", _group_table(groups, ev)]

    easy = ev["easyocr"]
    out += [
        "## Why EasyOCR was removed\n",
        "Version 0.4 chose between Tesseract, EasyOCR and a merge of both. EasyOCR needs torch, "
        f"about 2 GB installed. On the {easy['n']} train and validation images of the 0.4 run, "
        f"an oracle over those three actions reaches CER {easy['v04_actions']:.3f}. An oracle "
        f"over the pipelines shipped now reaches {easy['shipped']:.3f}, and adding EasyOCR to "
        f"them as one more choice gives {easy['with_easyocr']:.3f}.\n",
    ]

    size_note = (
        f"{card['unrouted']} images of this corpus are that large and are scored that way."
        if card["unrouted"]
        else "No such image is in this corpus."
    )
    val = corrector["val"]
    decision = (
        "It stays on by default."
        if corrector["decision"] == "keep"
        else "The interval does not lie below zero, so the correction was removed."
    )
    out += [
        "## Text correction\n",
        "Version 0.4 passed every result through a spelling corrector. Measured on the "
        f"{val['n']} validation images for the shipped policy: CER {_ci(val['cer_raw'])} without "
        f"it and {_ci(val['cer_corrected'])} with it, a paired difference of "
        f"{_signed(val['delta'])}; it changes the text of {val['changed_share'] * 100:.0f}% of "
        f"images. {decision}\n",
        "## Limitations\n",
        "- The degraded images are synthetic. Noise and heavy JPEG are rarer in real captures "
        "than in this corpus; small text and dark themes are common.\n"
        "- The browser pages are laid out by a real browser but are not captures of real "
        "applications, and no manually transcribed screenshots are included.\n"
        "- English and Russian only, with Tesseract configured for both.\n"
        "- Times are from one machine. The time weight trades error for seconds as measured "
        "there.\n"
        "- The app does not route an image larger than 2 megapixels: it runs "
        f"`{LARGE_IMAGE.name}` on it, which upscales only small images. {size_note}\n",
        "## Reproduce\n",
        "```bash\n"
        "venv/bin/python benchmarks/browser.py          # browser pages (needs Chrome)\n"
        "venv/bin/python benchmarks/run_eval.py         # every pipeline over the corpus\n"
        "venv/bin/python benchmarks/run_eval.py --timing  # timings, on an idle machine\n"
        "venv/bin/python benchmarks/train_router.py     # selection, evaluation, packaged model\n"
        "venv/bin/python benchmarks/report.py           # this file and its figure\n"
        "```\n",
        "The 0.4 rows come from `benchmarks/legacy_v04.json`, frozen from version 0.4.0.\n",
    ]
    if corrector["decision"] != "keep":
        out.append(
            "The text-correction numbers come from `benchmarks/corrector_eval.json`, measured "
            "before the corrector and its measurement script were removed.\n"
        )
    return "\n".join(out)


def main() -> None:
    ev = json.loads(_EVAL.read_text())
    corrector = json.loads(_CORRECTOR.read_text())
    environment = json.loads(_ENVIRONMENT.read_text()) if _ENVIRONMENT.exists() else None
    _IMG.mkdir(parents=True, exist_ok=True)
    if ev["shipped"] != "static":
        _plot_curve(ev, _IMG / "cer_time.png")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _REPORT.write_text(render(ev, corrector, environment, _git_commit(), generated))
    print(f"Wrote {_REPORT}")


if __name__ == "__main__":
    main()
