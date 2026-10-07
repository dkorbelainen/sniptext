"""Write docs/benchmark.md and its figures from the benchmark outputs.

Every number in the report comes from router_eval.json or from the calibration
analyses run here; nothing is typed in by hand.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.calib_merge import evaluate as calib_merge_eval
from benchmarks.calibration import calibrate, reliability

_HERE = Path(__file__).resolve().parent
_EVAL = _HERE / "router_eval.json"
_WORD_CONF = _HERE / "word_conf.json"
_REPO_ROOT = _HERE.parent
_REPORT = _REPO_ROOT / "docs" / "benchmark.md"
_IMG = _REPO_ROOT / "docs" / "img"

_LABELS = {
    "always_tesseract": "Always Tesseract",
    "always_easyocr": "Always EasyOCR",
    "always_merge": "Always merge",
    "legacy_rules": "Previous selector (rules)",
    "router_pre_ocr": "Router, image features",
    "router_cascade": "Router, cascade",
    "oracle": "Oracle (lower bound)",
}
_ACTION_LABELS = {"tesseract": "Tesseract", "easyocr": "EasyOCR", "merge": "merge"}


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


def verdict(summary: dict, best_static: str) -> str:
    """One sentence that says only what the paired interval supports."""
    mean, low, high = summary["delta"]
    name = _LABELS[summary["policy"]]
    static = _ACTION_LABELS[best_static]
    if high < 0:
        relation = f"lower than always running {static} by {-mean:.3f}"
    elif low > 0:
        relation = f"higher than always running {static} by {mean:.3f}"
    else:
        relation = f"not distinguishable from always running {static}"
    return (
        f"{name}: CER {_ci(summary['cer'])}, {relation} "
        f"(paired difference {_signed(summary['delta'])}, 95% interval over texts)."
    )


def _slice_section(ev: dict, key: str) -> list:
    """Verdict and table of one slice; nothing when the run has no such slice."""
    if key not in ev["slices"]:
        return []
    summaries = ev["slices"][key]
    router = next(s for s in summaries if s["policy"] == f"router_{ev['shipped']}")
    return [verdict(router, ev["best_static"]) + "\n", _slice_table(summaries, ev["shipped"])]


def _shipping_rule(ev: dict) -> str:
    shipped = ev["shipped"]
    other = "cascade" if shipped == "pre_ocr" else "pre_ocr"
    mine, theirs = ev["policies"][shipped]["oof"], ev["policies"][other]["oof"]
    return (
        "Of the two routers the one with the lower out-of-fold CER ships, or the faster one when "
        f"they are within 0.005: {_LABELS['router_' + shipped]} has {mine['cer']:.3f} at "
        f"{mine['time'] * 1000:.0f} ms per image, {_LABELS['router_' + other]} "
        f"{theirs['cer']:.3f} at {theirs['time'] * 1000:.0f} ms."
    )


def _curve_span(ev: dict, policy: str) -> str:
    curve = ev["policies"][policy]["curve"]
    times = [point[2] * 1000 for point in curve]
    cers = [point[1] for point in curve]
    return (
        f"{_LABELS['router_' + policy]} spans {min(times):.0f} to {max(times):.0f} ms and CER "
        f"{min(cers):.3f} to {max(cers):.3f}"
    )


def _slice_table(summaries: list, shipped: str) -> str:
    lines = [
        "| Policy | CER, clipped at 1 | CER, raw | Median | Regret to oracle "
        "| Difference to best static | Tesseract / EasyOCR / merge | Time, ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        label = _LABELS[s["policy"]]
        if s["policy"] == f"router_{shipped}":
            label = f"**{label} (shipped)**"
        share = " / ".join(f"{s['share'][a] * 100:.0f}%" for a in ("tesseract", "easyocr", "merge"))
        delta = _signed(s["delta"]) if "delta" in s else ""
        lines.append(
            f"| {label} | {_ci(s['cer'])} | {s['cer_unclipped']:.3f} | {s['cer_median']:.3f} "
            f"| {_ci(s['regret'])} | {delta} | {share} | {s['time'] * 1000:.0f} |"
        )
    return "\n".join(lines) + "\n"


def _breakdown_table(groups: dict) -> str:
    policies = [k for k in next(iter(groups.values())) if k != "n"]
    lines = [
        "| Group | n | " + " | ".join(_LABELS[p] for p in policies) + " |",
        "|---|---|" + "---|" * len(policies),
    ]
    for group, values in groups.items():
        cells = " | ".join(f"{values[p]:.3f}" for p in policies)
        lines.append(f"| {group} | {values['n']} | {cells} |")
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
    routers = (("pre_ocr", "Router, image features"), ("cascade", "Router, cascade"))
    for (policy, label), color in zip(routers, _SERIES):
        curve = sorted(ev["policies"][policy]["curve"], key=lambda point: point[2])
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
            label=label,
        )
    for summary in ev["slices"]["test"]:
        if summary["policy"].startswith("always_") or summary["policy"] == "oracle":
            point = (summary["time"] * 1000, summary["cer"][0])
            ax.plot(
                *point,
                linestyle="none",
                marker="D",
                markersize=6,
                color=_INK_SECONDARY,
                markeredgecolor=_SURFACE,
                markeredgewidth=1.5,
            )
            # The oracle sits under the router curves; its label goes below the marker.
            below = summary["policy"] == "oracle"
            ax.annotate(
                _LABELS[summary["policy"]],
                point,
                textcoords="offset points",
                xytext=(6, -6 if below else 6),
                va="top" if below else "baseline",
                fontsize=8,
                color=_INK,
            )
    ax.set_ylim(bottom=0)
    _finish(
        plt, fig, ax, path, "Mean time per image, ms (EasyOCR on GPU)", "Mean CER on held-out texts"
    )


def _plot_reliability(bins: list, path: Path) -> None:
    """Word accuracy against stated confidence, pooled over both engines."""
    plt, fig, ax = _new_axes((4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=_INK_SECONDARY, linewidth=1, label="Perfect calibration")
    ax.plot(
        [b["conf"] for b in bins],
        [b["accuracy"] for b in bins],
        color=_SERIES[0],
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=_SURFACE,
        markeredgewidth=1.5,
        label="Engines",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _finish(plt, fig, ax, path, "Stated word confidence", "Share of words correct")


def render(ev: dict, calib: dict, merge: dict, commit: str, generated: str) -> str:
    card = ev["dataset"]
    shipped = ev["shipped"]
    ship = ev["policies"][shipped]
    test = {s["policy"]: s for s in ev["slices"]["test"]}
    weak = ev["weak_labels"]["synthetic"]
    splits = ", ".join(f"{name} {count}" for name, count in card["by_split"].items())

    out = [
        "# OCR engine router benchmark\n",
        f"Generated: {generated}. Commit: `{commit}`. Engines run with `{card['language']}`.\n",
        "## Task\n",
        "For each screenshot SnipText picks one of three actions: Tesseract, EasyOCR, or both "
        "merged by word confidence. The router predicts the character error rate (CER) of each "
        "action and takes the one minimising predicted CER plus a time weight times the action's "
        "seconds. Two routers are compared: one that sees only image statistics before any OCR, "
        "and a cascade that runs Tesseract first and also sees its word confidences.\n",
        "CER is the edit distance to the ground truth over its length, whitespace-normalised, "
        "case kept. Averages use CER clipped at 1 so that one garbage output does not dominate; "
        "the raw mean is shown next to it. Intervals are 95% bootstrap intervals that resample "
        "texts, not images, because renders of one text are not independent.\n",
        "## Data\n",
        f"{card['n']} images: {splits}. {card['texts']} distinct texts "
        f"({card['by_lang']['en']} English and {card['by_lang']['ru']} Russian renders; "
        f"prose {card['by_content']['prose']}, code {card['by_content']['code']}, "
        f"interface strings {card['by_content']['ui']}). Sentences come from UD English-EWT and "
        "UD Russian-GSD, code from the CPython 3.12 standard library. Each text is rendered "
        "several times with a random font, size from 11 to 30 px, one of six colour schemes and "
        "up to two degradations out of blur, noise, JPEG, rescaling and low contrast.\n",
        "A text belongs to exactly one of train, validation and test. Two font families appear "
        "only in the unseen-font slice. The receipts (SROIE) are never used for fitting or "
        "selection.\n",
        "## Result on held-out texts\n",
        *_slice_section(ev, "test"),
        f"The shipped policy is `{shipped}`: {_spec(ship['spec'])}, time weight "
        f"{ship['time_weight']:.3f}. It is scored through `sniptext.router.Router`, the class the "
        'app uses, fitted from the table packaged with the app. "Best static" is always running '
        f"{_ACTION_LABELS[ev['best_static']]}, chosen on train and validation.\n",
        _shipping_rule(ev) + "\n",
        "## Unseen fonts\n",
        *_slice_section(ev, "unseen_font"),
        "## Out-of-domain receipts\n",
        "Photographed receipts are unlike anything in the training data. This slice shows what "
        "the router does outside its domain.\n",
        *_slice_section(ev, "ood"),
        "## Accuracy against time\n",
        "![CER against time](img/cer_time.png)\n",
        "Each line traces one router as the time weight grows from 0. Times are measured on this "
        "machine with EasyOCR on a GPU.\n",
        f"{_curve_span(ev, 'pre_ocr')}; {_curve_span(ev, 'cascade')}. The cascade runs Tesseract "
        "on every image, so its time cannot fall below Tesseract's.\n",
    ]

    if ev["cpu"] is None:
        out.append("CPU timing was not measured in this run.\n")
    else:
        cpu = ev["cpu"]
        lines = ["| Policy | CER | Time, ms | Time weight |", "|---|---|---|---|"]
        for action, values in cpu["always"].items():
            lines.append(
                f"| Always {_ACTION_LABELS[action]} | {values['cer']:.3f} "
                f"| {values['time'] * 1000:.0f} | |"
            )
        for policy, values in cpu["policies"].items():
            lines.append(
                f"| {_LABELS['router_' + policy]} | {values['cer']:.3f} "
                f"| {values['time'] * 1000:.0f} | {values['time_weight']:.3f} |"
            )
        out += [
            f"On a CPU EasyOCR is {cpu['ratio']:.1f} times slower (measured on a sample of "
            "images). With EasyOCR times scaled by that factor and the time weight re-chosen by "
            "the same rule, on held-out texts:\n",
            "\n".join(lines) + "\n",
        ]

    out += [
        "## Model selection\n",
        "Candidates are compared by grouped 5-fold cross-validation over train and validation "
        "texts (a text is never in both the fitting and the predicted fold), on the regret of the "
        "resulting policy to the oracle. The default time weight is the largest one that keeps "
        "out-of-fold CER within 0.005 of the accuracy-only policy.\n",
    ]
    for policy in ("pre_ocr", "cascade"):
        p = ev["policies"][policy]
        out += [
            f"### {_LABELS['router_' + policy]}\n",
            _selection_table(p["selection"]),
            f"Out-of-fold at its default time weight {p['time_weight']:.3f}: CER "
            f"{p['oof']['cer']:.3f}, {p['oof']['time'] * 1000:.0f} ms per image.\n",
        ]
    out += [
        f"Tesseract is called through `image_to_data` (`{ev['tesseract_call']}` call); its text "
        f"differs from `image_to_string` by {ev['tesseract_call_gap']:+.3f} mean CER on train and "
        "validation.\n",
        "## Features\n",
        f"Inputs of the shipped router. Extraction takes {ev['feature_time_mean'] * 1000:.1f} ms "
        "per image on average.\n",
        _feature_table(ship),
        "## Breakdown\n",
        "Mean clipped CER on held-out texts.\n",
    ]
    for key, title in (
        ("degradation", "By degradation"),
        ("theme", "By colour scheme"),
        ("lang", "By language"),
        ("content", "By content"),
    ):
        out += [f"### {title}\n", _breakdown_table(ev["breakdown"][key])]

    legacy = test["legacy_rules"]
    out += [
        "## The previous selector\n",
        "Before this router SnipText chose between Tesseract and the merge with fixed thresholds "
        "on image statistics and a classifier fitted on hand-made feature ranges. On held-out "
        f"texts it scores CER {_ci(legacy['cer'])} and picks the merge for "
        f"{legacy['share']['merge'] * 100:.0f}% of images.\n",
        "It also retrained itself on a label derived from a text-quality score of the two "
        "outputs. Compared with the action that really had the lower CER, that label agrees on "
        f"{weak['accuracy'] * 100:.0f}% of {weak['n']} synthetic images (Cohen's kappa "
        f"{weak['kappa']:.2f}); it names the merge for {weak['weak_merge_share'] * 100:.0f}% of "
        f"images while the merge is better on {weak['true_merge_share'] * 100:.0f}%. Online "
        "retraining was removed for that reason.\n",
        "## Confidence calibration\n",
        "Per-word confidence against word correctness. ECE is the population-weighted gap "
        "between confidence and accuracy over 10 bins. The calibrated column refits an isotonic "
        "map on 70% of the images and measures on the rest.\n",
        "| Domain | Words | Accuracy | Mean confidence | ECE raw | ECE calibrated |",
        "|---|---|---|---|---|---|",
    ]
    for domain, values in calib.items():
        out.append(
            f"| {domain} | {values['n']} | {values['accuracy']:.3f} | {values['mean_conf']:.3f} "
            f"| {values['ece_raw']:.3f} | {values['ece_cal']:.3f} |"
        )
    out += [
        "\n![Reliability diagram](img/reliability.png)\n",
        "The merge replayed on held-out texts with three ways of resolving disagreements:\n",
        "| Domain | Images | CER, text heuristic | CER, raw confidence | CER, calibrated |",
        "|---|---|---|---|---|",
    ]
    for domain, values in merge.items():
        out.append(
            f"| {domain} | {values['n_test']} | {values['cer_heuristic']:.3f} "
            f"| {values['cer_rawconf']:.3f} | {values['cer_calibrated']:.3f} |"
        )
    out += [
        "\n## Limitations\n",
        "- The images are rendered, not captured. Real screenshots have anti-aliasing, mixed "
        "fonts, icons and layouts this corpus does not cover; no manually transcribed "
        "screenshots are included.\n"
        "- English and Russian only, with engines configured for both. A single-language setup "
        "may rank the engines differently.\n"
        "- Times are from one machine. The time weight trades error for seconds as measured "
        "there.\n"
        "- EasyOCR reports one confidence per detected line; the word-level merge treats it as "
        "the confidence of every word in the line.\n",
        "## Reproduce\n",
        "```bash\n"
        "venv/bin/python benchmarks/run_eval.py      # engines over the corpus, resumable\n"
        "venv/bin/python benchmarks/train_router.py  # selection, evaluation, packaged table\n"
        "venv/bin/python benchmarks/report.py        # this file and its figures\n"
        "```\n",
    ]
    return "\n".join(out)


def main():
    ev = json.loads(_EVAL.read_text())
    calib = calibrate()
    merge = calib_merge_eval()
    _IMG.mkdir(parents=True, exist_ok=True)
    _plot_curve(ev, _IMG / "cer_time.png")
    _plot_reliability(reliability(json.loads(_WORD_CONF.read_text())), _IMG / "reliability.png")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _REPORT.write_text(render(ev, calib, merge, _git_commit(), generated))
    print(f"Wrote {_REPORT}")


if __name__ == "__main__":
    main()
