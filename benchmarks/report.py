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
    (
        "confirm",
        "Fresh texts",
        "No six-word run shared with any other slice. Generated after the shipped policy was "
        "fixed.\n",
    ),
    ("test", "Held-out texts", ""),
    ("unseen_font", "Unseen fonts", "Two font families used in no other slice.\n"),
    (
        "browser",
        "Browser pages",
        "Held-out texts rendered by headless Chrome: light and dark themes, device scale 1, 1.5 "
        "and 2, no degradation.\n",
    ),
    ("ood", "Receipts", "Photographed receipts (SROIE).\n"),
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


def _bound(value: float) -> str:
    """Three signed decimals; five when a non-zero bound would otherwise print as zero."""
    text = f"{value:+.3f}"
    return f"{value:+.5f}" if value != 0 and float(text) == 0 else text


def _signed(triple) -> str:
    mean, low, high = triple
    return f"{mean:+.3f} [{_bound(low)}, {_bound(high)}]"


def _relation(delta) -> str:
    """What a paired interval supports, as words to put before the thing compared with."""
    if delta[2] < 0:
        return "lower than"
    if delta[1] > 0:
        return "higher than"
    return "not distinguishable from"


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


def _slice_table(summaries: list, ev: dict) -> str:
    lines = [
        "| Policy | CER | Difference to best static | Time, ms |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        label = _label(s["policy"], ev)
        if s["policy"] == _shipped_label(ev):
            label = f"**{label} (shipped)**"
        delta = _signed(s["delta"]) if "delta" in s else ""
        lines.append(f"| {label} | {_ci(s['cer'])} | {delta} | {s['time'] * 1000:.0f} |")
    return "\n".join(lines) + "\n"


def _slice_section(ev: dict, key: str) -> list:
    """Table of one slice and the shipped policy's paired differences to version 0.4."""
    out = [_slice_table(ev["slices"][key], ev)]
    against = [
        f"against {name}: {_signed(deltas[key])}"
        for name, deltas in (
            (f"0.4 Tesseract (`{V04.name}`)", ev["per_slice_delta_v04_tesseract"]),
            ("the 0.4 router", ev["per_slice_delta_v04"]),
        )
        if key in deltas
    ]
    if against:
        out.append(f"Shipped policy {'; '.join(against)}.\n")
    return out


def _criteria(ev: dict) -> str:
    c = ev["criteria"]

    def line(text: str, met: bool | None, delta) -> str:
        state = "not measured" if met is None else "met" if met else "not met"
        numbers = f" ({_signed(delta)})" if delta else ""
        return f"{text}: **{state}**{numbers}."

    lines = [
        line("1. Lower CER on held-out texts than the best static pipeline",
             c["router_beats_best_static"], c["test_delta"]),
        line("2. Not worse than the 0.4 router, which needs EasyOCR",
             c["not_worse_than_v04"], c["v04_delta"]),
        line("3. Not worse than the best static pipeline on browser pages",
             c["browser_not_worse"], c["browser_delta"]),
    ]  # fmt: skip
    if c["confirm_delta"]:
        lines.append(
            line(
                "4. Lower CER on fresh texts than the best static pipeline",
                c["confirm_beats_best_static"],
                c["confirm_delta"],
            )  # fmt: skip
        )
    return "\n".join(lines) + "\n"


def _rule_change(ev: dict) -> list:
    """Why the policy that ships is not the one the first rule chose; nothing when it is."""
    c = ev["criteria"]
    first = c["preregistered_policy"]
    if ev["shipped"] == "static" or first == ev["shipped"]:
        return []
    label = f"router_{first}"
    out = (
        "The rule fixed before the measurement (within 0.005 out-of-fold CER the faster router "
        f"ships) chose {_FIXED[label]}. Its held-out CER is "
        f"{_relation(c['preregistered_test_delta'])} the best static pipeline's "
        f"({_signed(c['preregistered_test_delta'])})"
    )
    quiet = ev["noise_split"].get("test", {}).get("without added noise", {})
    if label in quiet.get("delta_static", {}):
        delta = quiet["delta_static"][label]
        out += f" and, on images without added noise, {_relation(delta)} it ({_signed(delta)})"
    return [
        out + ". The rule was replaced after the held-out results were seen, so the held-out "
        "numbers of the shipped router are not a clean estimate. The fresh texts were generated "
        "afterwards.\n"
    ]


def _noise_tables(ev: dict, name: str) -> list:
    split = ev["noise_split"][name]
    labels = list(next(iter(split.values()))["cer"])
    means = [
        "| Group | n | " + " | ".join(_label(label, ev) for label in labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    pairs = [
        "| Group | Router | Difference to best static | Difference to 0.4 Tesseract |",
        "|---|---|---|---|",
    ]
    for group, values in split.items():
        cells = " | ".join(f"{values['cer'][label]:.3f}" for label in labels)
        means.append(f"| {group} | {values['n']} | {cells} |")
        for router, delta in values["delta_static"].items():
            old = values["delta_v04"].get(router)
            pairs.append(
                f"| {group} | {_FIXED[router]} | {_signed(delta)} | {_signed(old) if old else ''} |"
            )
    out = ["\n".join(means) + "\n"]
    if len(pairs) > 2:
        out.append("\n".join(pairs) + "\n")
    return out


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


def _selection_table(ev: dict) -> str:
    """The best candidate of each kind per router."""
    lines = [
        "| Router | Best candidate | Out-of-fold CER | Regret to oracle |",
        "|---|---|---|---|",
    ]
    for policy, values in ev["policies"].items():
        best = {}
        for entry in sorted(values["selection"], key=lambda e: -e["oof_regret"]):
            best[entry["spec"]["kind"]] = entry
        for entry in sorted(best.values(), key=lambda e: e["oof_regret"]):
            lines.append(
                f"| {_FIXED['router_' + policy]} | {_spec(entry['spec'])} "
                f"| {entry['oof_cer']:.3f} | {entry['oof_regret']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def _feature_note(policy: dict) -> str:
    names = policy["feature_names"]
    top = sorted(names, key=lambda n: -policy["importance"][n])[:3]
    shuffled = ", ".join(f"`{name}` {policy['importance'][name]:+.3f}" for name in top)
    return (
        f"The shipped router has {len(names)} inputs. Removing any one changes out-of-fold CER "
        f"by at most {max(policy['ablation'].values()):+.3f}. Shuffling one on held-out texts "
        f"changes CER most for {shuffled}.\n"
    )


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
    loads = [sample[0] for timing_pass in ev["loadavg"] for sample in timing_pass]
    chrome = f" Browser: {environment['chrome']}." if environment else ""

    out = [
        "# Benchmark\n",
        f"Generated: {generated}. Commit: `{commit}`. Tesseract runs with `{card['language']}`.\n",
        "## Method\n",
        "A pipeline is a preprocessing chain plus a Tesseract page-segmentation mode. A router "
        "predicts the character error rate (CER) of each shipped pipeline and takes the lowest "
        "predicted CER plus a time weight times the pipeline's seconds. Two routers are "
        "compared: one sees image statistics only; the cascade runs the default pipeline first "
        "and also sees its word confidences. An empty result falls back to the default "
        "pipeline.\n",
        "CER is the edit distance over the ground-truth length, whitespace-normalised, clipped "
        "at 1. Intervals are 95% bootstrap intervals over texts. Differences are paired.\n",
        "## Data\n",
        f"{card['n']} images from {card['texts']} texts: {splits}. English "
        f"({card['by_lang']['en']}) and Russian ({card['by_lang']['ru']}) renders of prose "
        f"({card['by_content']['prose']}), code ({card['by_content']['code']}) and interface "
        f"strings ({card['by_content']['ui']}) from UD English-EWT, UD Russian-GSD and CPython "
        "3.12. Random font, 11 to 30 px, six colour schemes, up to two degradations out of blur, "
        "noise, JPEG, rescaling and low contrast.\n",
        "Train, validation and test share no text. Receipts, browser pages and fresh texts are "
        f"not used for fitting or selection.{chrome}\n",
        "## Shipped policy\n",
    ]

    if routed:
        ship = ev["policies"][ev["shipped"]]
        other = "cascade" if ev["shipped"] == "pre_ocr" else "pre_ocr"
        mine, theirs = ship["oof"], ev["policies"][other]["oof"]
        out += [
            f"`{ev['shipped']}` over {actions}: {_spec(ship['spec'])}, time weight "
            f"{ship['time_weight']:.3f}.\n",
            f"Out-of-fold CER: {_FIXED['router_' + ev['shipped']]} {mine['cer']:.3f} at "
            f"{mine['time'] * 1000:.0f} ms per image, {_FIXED['router_' + other]} "
            f"{theirs['cer']:.3f} at {theirs['time'] * 1000:.0f} ms. The lower one ships.\n",
            *_rule_change(ev),
        ]
    else:
        out.append(f"No router ships: `{ev['best_static']}` runs on every image.\n")
    out += [
        "Criteria, with paired differences. 1 to 3 were fixed before the first measurement, 4 "
        "before the fresh texts were read.\n"
        if ev["criteria"]["confirm_delta"]
        else "Criteria fixed before the measurement, with paired differences.\n",
        _criteria(ev),
        "## Results\n",
        "CER with its interval, paired difference to the best static pipeline, mean time per "
        "image. The 0.4 router's times are from a GPU run and not comparable.\n",
    ]
    for key, title, intro in _SLICES:
        if key in ev["slices"]:
            out += [f"### {title}\n", *([intro] if intro else []), *_slice_section(ev, key)]

    if ev["noise_split"]:
        out += [
            "## Added noise\n",
            "Mean CER per group, then paired differences. The split was not planned before the "
            "measurement.\n",
        ]
        for name, heading in (("confirm", "Fresh texts"), ("test", "Held-out texts")):
            if name in ev["noise_split"]:
                out += [f"### {heading}\n", *_noise_tables(ev, name)]

    degradation = ev["breakdown"].get("test", {}).get("degradation")
    if degradation:
        out += ["## By degradation\n", "Held-out texts.\n", _group_table(degradation, ev)]

    out.append("## Time\n")
    if routed:
        out += [
            "![CER against time](img/cer_time.png)\n",
            "One line per router as the time weight grows from 0.\n",
        ]
    out.append(
        f"One process, one machine, one-minute load average {min(loads):.1f} to "
        f"{max(loads):.1f}. Feature extraction ({ev['feature_time_mean'] * 1000:.1f} ms per "
        "image) is not included.\n"
    )

    out += [
        "## Pipeline selection\n",
        f"{len(ev['pool'])} candidates; `{V04.name}` is the pipeline of 0.4.\n",
        _pool_table(ev),
        "Greedy on train and validation texts: start from the best static pipeline, add the "
        "candidate that lowers out-of-fold CER most, stop below a gain of 0.003 or at four "
        "pipelines.\n",
        _steps_table(ev),
    ]

    if routed:
        out += [
            "## Model selection\n",
            "Grouped 5-fold cross-validation over train and validation texts, ranked by regret "
            "to the oracle. The time weight is the largest that keeps out-of-fold CER within "
            "0.005 of the accuracy-only policy.\n",
            _selection_table(ev),
            _feature_note(ev["policies"][ev["shipped"]]),
        ]

    easy = ev["easyocr"]
    val = corrector["val"]
    decision = "It stays on by default." if corrector["decision"] == "keep" else "It was removed."
    size_note = (
        f"{card['unrouted']} images here are that large."
        if card["unrouted"]
        else "No image here is that large."
    )
    out += [
        "## Removed components\n",
        f"EasyOCR (needs torch, about 2 GB): on the {easy['n']} train and validation images of "
        f"the 0.4 run, an oracle over the three actions of 0.4 reaches CER "
        f"{easy['v04_actions']:.3f}, an oracle over the shipped pipelines {easy['shipped']:.3f}, "
        f"and {easy['with_easyocr']:.3f} with EasyOCR added.\n",
        f"Spelling corrector, {val['n']} validation images: CER {_ci(val['cer_raw'])} without "
        f"it, {_ci(val['cer_corrected'])} with it, paired difference {_signed(val['delta'])}; it "
        f"changes {val['changed_share'] * 100:.0f}% of texts. {decision}\n",
        "## Limitations\n",
        "- Degradations are synthetic; noise and heavy JPEG are rarer in real captures.\n"
        "- No captures of real applications and no manually transcribed screenshots.\n"
        "- English and Russian only.\n"
        "- Times are from one machine.\n"
        f"- Train and validation images are at most {card['dev_max_pixels'] / 1e6:.3f} megapixels "
        "and pipeline costs are means over them. Images up to 2 megapixels are routed, where "
        "upscaling costs more (see the receipt times).\n"
        "- The candidate pool followed a finding on the held-out slice of the 0.4 run (the "
        "second engine helped mostly on noisy images), so that slice is not blind to the pool.\n"
        f"- Images above 2 megapixels run `{LARGE_IMAGE.name}` unrouted; the rule was set after "
        f"the receipt timings were seen. {size_note}\n",
        "## Reproduce\n",
        "```bash\n"
        "venv/bin/python benchmarks/browser.py            # browser pages (needs Chrome)\n"
        "venv/bin/python benchmarks/run_eval.py           # every pipeline over the corpus\n"
        "venv/bin/python benchmarks/run_eval.py --timing  # timings, on an idle machine\n"
        "venv/bin/python benchmarks/train_router.py       # selection, evaluation, packaged model\n"
        "venv/bin/python benchmarks/report.py             # this file and its figure\n"
        "```\n",
        "0.4 rows: `benchmarks/legacy_v04.json`, frozen from 0.4.0.\n",
    ]
    if corrector["decision"] != "keep":
        out.append(
            "Corrector numbers: `benchmarks/corrector_eval.json`, measured before its removal.\n"
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
