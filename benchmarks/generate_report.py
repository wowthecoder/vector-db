#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard from Google Benchmark JSON."""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TIME_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
METRIC_COLORS = {
    "L2": "#2563eb",
    "Dot": "#ea580c",
    "Cosine": "#059669",
    "Batch": "#2563eb",
    "Repeated singles": "#dc2626",
    "Insert": "#7c3aed",
    "Save": "#0891b2",
    "Load": "#d97706",
}
FALLBACK_COLORS = ("#2563eb", "#ea580c", "#059669", "#7c3aed", "#0891b2")
PARAMETER_ORDER = {
    "insert": ("count", "dimension"),
    "save": ("count", "dimension"),
    "load": ("count", "dimension"),
    "search": ("count", "dimension", "top_k"),
    "lsh_search": (
        "count",
        "dimension",
        "top_k",
        "num_tables",
        "num_bits",
        "num_candidates",
        "query_count",
    ),
    "batch_search": ("count", "dimension", "query_count", "top_k"),
    "repeated_search": ("count", "dimension", "query_count", "top_k"),
}
OPERATION_NAMES = {
    "insert": "Insert",
    "search": "Single search",
    "lsh_search": "LSH search",
    "batch_search": "Batch search",
    "repeated_search": "Repeated single search",
    "save": "Save",
    "load": "Load",
    "unknown": "Unknown",
}


class ReportError(ValueError):
    """Raised when an input cannot be converted into a report."""


@dataclass
class Result:
    name: str
    operation: str
    metric: str | None
    params: dict[str, int | float | str]
    real_ms: float
    cpu_ms: float
    items_per_second: float | None
    bytes_per_second: float | None
    recall_at_k: float | None
    lsh_build_ms: float | None
    index_payload_bytes: float | None
    repetitions: int
    cv_percent: float | None
    threads: int
    source: str


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def parse_scalar(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_benchmark_name(name: str) -> tuple[str, str | None, dict[str, int | float | str]]:
    parts = name.split("/")
    function = parts[0]
    metric_match = re.search(r"<[^>]*::([^>:]+)>", function)
    metric = metric_match.group(1) if metric_match else None

    if function.startswith("BM_RandomProjectionLshSearch"):
        operation = "lsh_search"
        metric = "Cosine"
    elif function.startswith("BM_CollectionBatchSearch"):
        operation = "batch_search"
    elif function.startswith("BM_RepeatedSingleSearch"):
        operation = "repeated_search"
    elif function.startswith("BM_CollectionSearch"):
        operation = "search"
    elif function.startswith("BM_CollectionInsert"):
        operation = "insert"
    elif function.startswith("BM_CollectionSave"):
        operation = "save"
    elif function.startswith("BM_CollectionLoad"):
        operation = "load"
    else:
        operation = "unknown"

    params: dict[str, int | float | str] = {}
    positional: list[int | float | str] = []
    for token in parts[1:]:
        if token in {"real_time", "manual_time"}:
            continue
        if ":" in token:
            key, value = token.split(":", 1)
            if key not in {"repeats", "threads", "min_time"}:
                params[key] = parse_scalar(value)
        elif token:
            positional.append(parse_scalar(token))

    for key, value in zip(PARAMETER_ORDER.get(operation, ()), positional):
        params.setdefault(key, value)
    return operation, metric, params


def _median_row(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], float | None, str]:
    aggregates = {row.get("aggregate_name"): row for row in rows if row.get("run_type") == "aggregate"}
    if "median" in aggregates:
        cv = finite_number(aggregates.get("cv", {}).get("real_time"))
        return dict(aggregates["median"]), None if cv is None else cv * 100.0, "median"

    raw = [row for row in rows if row.get("run_type", "iteration") == "iteration"]
    if raw:
        if len(raw) == 1:
            return dict(raw[0]), None, "single run"
        selected = copy.deepcopy(raw[0])
        for key in (
            "real_time",
            "cpu_time",
            "items_per_second",
            "bytes_per_second",
            "recall_at_k",
            "lsh_build_ms",
            "index_payload_bytes",
        ):
            values = [number for row in raw if (number := finite_number(row.get(key))) is not None]
            if values:
                selected[key] = statistics.median(values)
        real_values = [number for row in raw if (number := finite_number(row.get("real_time"))) is not None]
        cv = None
        if len(real_values) > 1 and statistics.mean(real_values) != 0:
            cv = statistics.stdev(real_values) / statistics.mean(real_values) * 100.0
        selected["repetitions"] = len(raw)
        return selected, cv, "computed median"

    if "mean" in aggregates:
        return dict(aggregates["mean"]), None, "mean fallback"
    raise ReportError("a benchmark group contains no usable timing row")


def load_results(path: Path) -> tuple[dict[str, Any], list[Result], list[str]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReportError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ReportError(f"invalid JSON in {path}: {error}") from error

    if not isinstance(document, dict) or not isinstance(document.get("benchmarks"), list):
        raise ReportError("input must be a Google Benchmark JSON object with a 'benchmarks' array")
    if not document["benchmarks"]:
        raise ReportError("the benchmark array is empty")

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    warnings: list[str] = []
    for index, row in enumerate(document["benchmarks"]):
        if not isinstance(row, dict):
            warnings.append(f"Skipped benchmark entry {index}: expected an object.")
            continue
        if row.get("error_occurred") or row.get("skipped"):
            message = row.get("error_message") or row.get("skip_message") or "benchmark was skipped"
            warnings.append(f"Skipped {row.get('name', f'entry {index}')}: {message}.")
            continue
        key = str(row.get("run_name") or row.get("name") or f"entry-{index}")
        groups[key].append(row)

    results: list[Result] = []
    for run_name, rows in groups.items():
        try:
            row, cv_percent, source = _median_row(rows)
        except ReportError as error:
            warnings.append(f"Skipped {run_name}: {error}.")
            continue
        real = finite_number(row.get("real_time"))
        cpu = finite_number(row.get("cpu_time"))
        unit = row.get("time_unit")
        if real is None or cpu is None or unit not in TIME_TO_MS:
            warnings.append(f"Skipped {run_name}: missing finite timing values or unsupported time unit.")
            continue
        operation, metric, params = parse_benchmark_name(run_name)
        results.append(
            Result(
                name=run_name,
                operation=operation,
                metric=metric,
                params=params,
                real_ms=real * TIME_TO_MS[unit],
                cpu_ms=cpu * TIME_TO_MS[unit],
                items_per_second=finite_number(row.get("items_per_second")),
                bytes_per_second=finite_number(row.get("bytes_per_second")),
                recall_at_k=finite_number(row.get("recall_at_k")),
                lsh_build_ms=finite_number(row.get("lsh_build_ms")),
                index_payload_bytes=finite_number(
                    row.get("index_payload_bytes")
                ),
                repetitions=int(row.get("repetitions", len(rows)) or len(rows)),
                cv_percent=cv_percent,
                threads=int(row.get("threads", 1) or 1),
                source=source,
            )
        )

    if not results:
        raise ReportError("no usable benchmark timing records were found")
    results.sort(key=lambda result: result.name)
    context = document.get("context") if isinstance(document.get("context"), dict) else {}
    return dict(context), results, warnings


def preferred_value(results: Sequence[Result], key: str, preferred: int) -> int | float | str:
    values = [result.params[key] for result in results if key in result.params]
    if preferred in values:
        return preferred
    if not values:
        return preferred
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], str(value)))[0]


def fmt_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.{decimals}f}k"
    if magnitude < 0.01:
        return f"{value:.4f}"
    return f"{value:.{decimals}f}"


def fmt_param(value: int | float | str) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def color_for(label: str, index: int) -> str:
    return METRIC_COLORS.get(label, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def nice_ticks(max_value: float, count: int = 5) -> tuple[float, list[float]]:
    if max_value <= 0:
        return 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]
    rough = max_value / count
    power = 10 ** math.floor(math.log10(rough))
    normalized = rough / power
    step_factor = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    step = step_factor * power
    ceiling = math.ceil(max_value / step) * step
    return ceiling, [index * step for index in range(int(round(ceiling / step)) + 1)]


def line_chart(
    chart_id: str,
    title: str,
    subtitle: str,
    x_values: Sequence[int | float | str],
    series: Mapping[str, Mapping[int | float | str, float]],
    x_label: str,
    y_label: str,
) -> str:
    if not x_values or not series:
        return empty_chart(title, "No matching benchmark cases were found.")
    width, height = 760, 350
    left, right, top, bottom = 74, 24, 34, 58
    plot_width, plot_height = width - left - right, height - top - bottom
    all_values = [value for points in series.values() for value in points.values()]
    ceiling, ticks = nice_ticks(max(all_values, default=0.0))

    def x_position(index: int) -> float:
        return left + (plot_width / 2 if len(x_values) == 1 else index * plot_width / (len(x_values) - 1))

    def y_position(value: float) -> float:
        return top + plot_height - value / ceiling * plot_height

    content = [
        f'<section class="chart-card"><h3>{escape(title)}</h3><p>{escape(subtitle)}</p>',
        f'<svg id="{escape(chart_id)}" class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<title>{escape(title)}</title>',
    ]
    for tick in ticks:
        y = y_position(tick)
        content.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        content.append(f'<text class="axis-label" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{escape(fmt_number(tick))}</text>')
    for index, value in enumerate(x_values):
        x = x_position(index)
        content.append(f'<text class="axis-label" x="{x:.2f}" y="{height-bottom+23}" text-anchor="middle">{escape(fmt_param(value))}</text>')
    content.append(f'<text class="axis-title" x="{left+plot_width/2:.2f}" y="{height-8}" text-anchor="middle">{escape(x_label)}</text>')
    content.append(f'<text class="axis-title" transform="translate(18 {top+plot_height/2:.2f}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>')

    legend_x = left
    for series_index, (label, points) in enumerate(series.items()):
        color = color_for(label, series_index)
        coordinates = [(x_position(index), y_position(points[value]), value, points[value]) for index, value in enumerate(x_values) if value in points]
        if coordinates:
            path = " ".join(("M" if position == 0 else "L") + f" {x:.2f} {y:.2f}" for position, (x, y, _, _) in enumerate(coordinates))
            dash = ' stroke-dasharray="7 5"' if label == "Repeated singles" else ""
            content.append(f'<path class="series" d="{path}" stroke="{color}"{dash}/>')
            for x, y, x_value, value in coordinates:
                content.append(
                    f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}">'
                    f'<title>{escape(label)} — {escape(x_label)} {escape(fmt_param(x_value))}: {escape(fmt_number(value, 3))} {escape(y_label)}</title></circle>'
                )
        content.append(f'<g transform="translate({legend_x} 15)"><line x1="0" y1="0" x2="22" y2="0" stroke="{color}" stroke-width="3"/><text class="legend" x="28" y="4">{escape(label)}</text></g>')
        legend_x += 44 + len(label) * 8
    content.append("</svg></section>")
    return "".join(content)


def bar_chart(
    chart_id: str,
    title: str,
    subtitle: str,
    categories: Sequence[str],
    series: Mapping[str, Sequence[float | None]],
    y_label: str,
) -> str:
    if not categories or not series:
        return empty_chart(title, "No matching benchmark cases were found.")
    width, height = 760, 350
    left, right, top, bottom = 74, 24, 34, 62
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [value for group in series.values() for value in group if value is not None]
    ceiling, ticks = nice_ticks(max(values, default=0.0))
    group_width = plot_width / max(len(categories), 1)
    bar_width = min(72.0, group_width * 0.72 / max(len(series), 1))

    content = [
        f'<section class="chart-card"><h3>{escape(title)}</h3><p>{escape(subtitle)}</p>',
        f'<svg id="{escape(chart_id)}" class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}"><title>{escape(title)}</title>',
    ]
    for tick in ticks:
        y = top + plot_height - tick / ceiling * plot_height
        content.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        content.append(f'<text class="axis-label" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{escape(fmt_number(tick))}</text>')
    for category_index, category in enumerate(categories):
        center = left + (category_index + 0.5) * group_width
        content.append(f'<text class="axis-label" x="{center:.2f}" y="{height-bottom+25}" text-anchor="middle">{escape(category)}</text>')
    content.append(f'<text class="axis-title" transform="translate(18 {top+plot_height/2:.2f}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>')

    legend_x = left
    series_items = list(series.items())
    total_width = bar_width * len(series_items)
    for series_index, (label, group) in enumerate(series_items):
        color = color_for(label, series_index)
        for category_index, value in enumerate(group):
            if value is None:
                continue
            center = left + (category_index + 0.5) * group_width
            x = center - total_width / 2 + series_index * bar_width
            bar_height = value / ceiling * plot_height
            y = top + plot_height - bar_height
            content.append(
                f'<rect class="bar" x="{x+2:.2f}" y="{y:.2f}" width="{max(bar_width-4, 1):.2f}" height="{bar_height:.2f}" fill="{color}">'
                f'<title>{escape(label)} — {escape(category)}: {escape(fmt_number(value, 3))} {escape(y_label)}</title></rect>'
            )
        content.append(f'<g transform="translate({legend_x} 15)"><rect x="0" y="-7" width="16" height="10" rx="2" fill="{color}"/><text class="legend" x="23" y="3">{escape(label)}</text></g>')
        legend_x += 42 + len(label) * 8
    content.append("</svg></section>")
    return "".join(content)


def empty_chart(title: str, message: str) -> str:
    return f'<section class="chart-card empty"><h3>{escape(title)}</h3><p>{escape(message)}</p></section>'


def result_matches(result: Result, **params: Any) -> bool:
    return all(result.params.get(key) == value for key, value in params.items())


def make_search_charts(results: Sequence[Result], canonical: Mapping[str, Any]) -> str:
    searches = [result for result in results if result.operation == "search"]
    definitions = [
        ("search-count", "Collection-size scaling", f"dimension={fmt_param(canonical['dimension'])}, top_k={fmt_param(canonical['top_k'])}", "count", {"dimension": canonical["dimension"], "top_k": canonical["top_k"]}),
        ("search-dimension", "Dimension scaling", f"count={fmt_param(canonical['count'])}, top_k={fmt_param(canonical['top_k'])}", "dimension", {"count": canonical["count"], "top_k": canonical["top_k"]}),
        ("search-top-k", "Result-count scaling", f"count={fmt_param(canonical['count'])}, dimension={fmt_param(canonical['dimension'])}", "top_k", {"count": canonical["count"], "dimension": canonical["dimension"]}),
    ]
    charts: list[str] = []
    for chart_id, title, subtitle, axis, fixed in definitions:
        selected = [result for result in searches if result_matches(result, **fixed) and axis in result.params]
        x_values = sorted({result.params[axis] for result in selected})
        series: dict[str, dict[Any, float]] = defaultdict(dict)
        for result in selected:
            series[result.metric or "Unspecified"][result.params[axis]] = result.real_ms
        charts.append(line_chart(chart_id, title, subtitle, x_values, series, axis.replace("_", " "), "wall ms"))
    return '<div class="chart-grid">' + "".join(charts) + "</div>"


def make_batch_section(results: Sequence[Result], canonical: Mapping[str, Any]) -> str:
    candidates = [
        result for result in results
        if result.operation in {"batch_search", "repeated_search"}
        and result_matches(result, count=canonical["count"], dimension=canonical["dimension"], top_k=canonical["top_k"])
        and "query_count" in result.params
    ]
    metrics = sorted({result.metric or "Unspecified" for result in candidates})
    charts: list[str] = []
    rows: list[str] = []
    for metric in metrics:
        metric_results = [result for result in candidates if (result.metric or "Unspecified") == metric]
        x_values = sorted({result.params["query_count"] for result in metric_results})
        series: dict[str, dict[Any, float]] = {"Batch": {}, "Repeated singles": {}}
        by_identity: dict[tuple[str, Any], Result] = {}
        for result in metric_results:
            method = "Batch" if result.operation == "batch_search" else "Repeated singles"
            query_count = result.params["query_count"]
            series[method][query_count] = result.real_ms / float(query_count)
            by_identity[(result.operation, query_count)] = result
        charts.append(line_chart(f"batch-{metric.lower()}", f"{metric} batch behavior", "Lower is better; latency normalized per query.", x_values, series, "queries per call", "wall ms/query"))
        for query_count in x_values:
            batch = by_identity.get(("batch_search", query_count))
            repeated = by_identity.get(("repeated_search", query_count))
            if not batch or not repeated:
                continue
            batch_per_query = batch.real_ms / float(query_count)
            repeated_per_query = repeated.real_ms / float(query_count)
            improvement = (repeated_per_query - batch_per_query) / repeated_per_query * 100.0 if repeated_per_query else 0.0
            classification = "positive" if improvement > 0 else "negative" if improvement < 0 else "neutral"
            rows.append(
                f"<tr><td>{escape(metric)}</td><td data-sort=\"{query_count}\">{escape(fmt_param(query_count))}</td>"
                f"<td data-sort=\"{batch_per_query}\">{escape(fmt_number(batch_per_query, 3))}</td>"
                f"<td data-sort=\"{repeated_per_query}\">{escape(fmt_number(repeated_per_query, 3))}</td>"
                f"<td class=\"delta {classification}\" data-sort=\"{improvement}\">{improvement:+.2f}%</td></tr>"
            )
    table = (
        '<div class="table-wrap"><table><thead><tr><th>Metric</th><th>Queries/call</th><th>Batch ms/query</th>'
        '<th>Repeated ms/query</th><th>Batch improvement</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return '<div class="chart-grid">' + "".join(charts) + "</div>" + table


def make_lsh_section(results: Sequence[Result]) -> str:
    lsh_results = sorted(
        (result for result in results if result.operation == "lsh_search"),
        key=lambda result: (
            result.params.get("count", 0),
            result.params.get("dimension", 0),
            result.params.get("top_k", 0),
            result.params.get("num_tables", 0),
            result.params.get("num_bits", 0),
            result.params.get("num_candidates", 0),
        ),
    )
    if not lsh_results:
        return empty_chart(
            "LSH recall and latency",
            "No random-projection LSH benchmark cases were found.",
        )

    rows: list[str] = []
    for result in lsh_results:
        recall_percent = (
            None
            if result.recall_at_k is None
            else result.recall_at_k * 100.0
        )
        payload_mib = (
            None
            if result.index_payload_bytes is None
            else result.index_payload_bytes / (1024.0 * 1024.0)
        )
        values = (
            result.params.get("count"),
            result.params.get("dimension"),
            result.params.get("top_k"),
            result.params.get("num_tables"),
            result.params.get("num_bits"),
            result.params.get("num_candidates"),
            result.real_ms,
            recall_percent,
            result.items_per_second,
            result.lsh_build_ms,
            payload_mib,
        )
        displays = (
            fmt_param(values[0]) if values[0] is not None else "—",
            fmt_param(values[1]) if values[1] is not None else "—",
            fmt_param(values[2]) if values[2] is not None else "—",
            fmt_param(values[3]) if values[3] is not None else "—",
            fmt_param(values[4]) if values[4] is not None else "—",
            fmt_param(values[5]) if values[5] is not None else "—",
            fmt_number(values[6], 4),
            "—" if values[7] is None else f"{values[7]:.2f}%",
            fmt_number(values[8]),
            fmt_number(values[9], 3),
            fmt_number(values[10], 3),
        )
        cells = "".join(
            f'<td data-sort="{escape("" if value is None else value)}">'
            f"{escape(display)}</td>"
            for value, display in zip(values, displays)
        )
        rows.append(f"<tr>{cells}</tr>")

    headers = (
        "Count",
        "Dimension",
        "Top K",
        "Tables",
        "Bits",
        "Candidates",
        "Wall ms/query",
        "Recall@K",
        "Queries/s",
        "Build ms",
        "Payload MiB",
    )
    header_html = "".join(
        f'<th tabindex="0" data-sortable="true">{escape(header)}</th>'
        for header in headers
    )
    return (
        '<div class="table-wrap"><table id="lsh-results"><thead><tr>'
        f"{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def make_write_charts(results: Sequence[Result]) -> str:
    inserts = sorted((result for result in results if result.operation == "insert"), key=lambda result: result.params.get("count", 0))
    insert_categories = [fmt_param(result.params.get("count", "?")) for result in inserts]
    insert_values = [result.items_per_second for result in inserts]
    insert_chart = bar_chart("insert-throughput", "Insertion throughput", "Google Benchmark reported items per second.", insert_categories, {"Insert": insert_values}, "vectors/s")

    persistence = [result for result in results if result.operation in {"save", "load"}]
    counts = sorted({result.params.get("count") for result in persistence if result.params.get("count") is not None})
    values: dict[str, list[float | None]] = {"Save": [], "Load": []}
    for count in counts:
        for operation, label in (("save", "Save"), ("load", "Load")):
            match = next((result for result in persistence if result.operation == operation and result.params.get("count") == count), None)
            values[label].append(match.real_ms if match else None)
    persistence_chart = bar_chart("persistence-latency", "Persistence latency", "Warm-cache wall latency; lower is better.", [fmt_param(count) for count in counts], values, "wall ms")
    return '<div class="chart-grid">' + insert_chart + persistence_chart + "</div>"


def canonical_results(results: Sequence[Result], canonical: Mapping[str, Any]) -> list[Result]:
    chosen: list[Result] = []
    for result in results:
        if result.operation == "search" and result_matches(result, count=canonical["count"], dimension=canonical["dimension"], top_k=canonical["top_k"]):
            chosen.append(result)
        elif result.operation in {"batch_search", "repeated_search"} and result_matches(result, count=canonical["count"], dimension=canonical["dimension"], query_count=canonical["query_count"], top_k=canonical["top_k"]):
            chosen.append(result)
        elif result.operation in {"insert", "save", "load"} and result.params.get("count") == canonical["count"] and result.params.get("dimension") == canonical["dimension"]:
            chosen.append(result)
    return sorted(chosen, key=lambda result: (result.operation, result.metric or ""))


def results_table(results: Sequence[Result], table_id: str, include_filter: bool) -> str:
    rows: list[str] = []
    for result in results:
        values = [
            OPERATION_NAMES.get(result.operation, result.operation),
            result.metric or "—",
            result.params.get("count"),
            result.params.get("dimension"),
            result.params.get("query_count"),
            result.params.get("top_k"),
            result.real_ms,
            result.cpu_ms,
            result.items_per_second,
            result.bytes_per_second,
            result.repetitions,
            result.cv_percent,
        ]
        cells: list[str] = []
        for index, value in enumerate(values):
            if value is None:
                display = "—"
                sort_value = ""
            elif index in {2, 3, 4, 5, 10}:
                display = fmt_param(value)
                sort_value = value
            elif index in {6, 7}:
                display = fmt_number(float(value), 3)
                sort_value = value
            elif index in {8, 9}:
                display = fmt_number(float(value))
                sort_value = value
            elif index == 11:
                display = f"{float(value):.2f}%"
                sort_value = value
            else:
                display = str(value)
                sort_value = display.lower()
            cells.append(f'<td data-sort="{escape(sort_value)}">{escape(display)}</td>')
        rows.append(f'<tr data-search="{escape(" ".join(str(value) for value in values if value is not None).lower())}">' + "".join(cells) + "</tr>")
    filter_box = f'<label class="filter">Filter results <input type="search" data-table-filter="{escape(table_id)}" placeholder="metric, operation, count…"></label>' if include_filter else ""
    headers = ("Operation", "Metric", "Count", "Dimension", "Queries", "Top K", "Wall ms", "CPU ms", "Items/s", "Effective bytes/s", "Repetitions", "Wall CV")
    header_html = "".join(f'<th tabindex="0" data-sortable="true">{escape(header)}</th>' for header in headers)
    return f'{filter_box}<div class="table-wrap"><table id="{escape(table_id)}"><thead><tr>{header_html}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def context_cards(context: Mapping[str, Any], results: Sequence[Result]) -> str:
    cards = [
        ("Date", context.get("date", "Unknown")),
        ("Host", context.get("host_name", "Unknown")),
        ("CPUs", context.get("num_cpus", "Unknown")),
        ("CPU frequency", f"{context.get('mhz_per_cpu')} MHz" if context.get("mhz_per_cpu") is not None else "Unknown"),
        ("Build", context.get("library_build_type", "Unknown")),
        ("Cases", len(results)),
        ("Repetitions", min(result.repetitions for result in results)),
        ("Load average", ", ".join(fmt_number(float(value), 2) for value in context.get("load_avg", [])) or "Unknown"),
    ]
    return '<div class="cards">' + "".join(f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>' for label, value in cards) + "</div>"


def quality_warnings(context: Mapping[str, Any], results: Sequence[Result], warnings: Sequence[str]) -> list[str]:
    messages = list(warnings)
    if min(result.repetitions for result in results) < 2:
        messages.insert(0, "This report contains only one repetition per case. Treat differences as trends, not statistically stable regressions or improvements.")
    if str(context.get("library_build_type", "")).lower() != "release":
        messages.append("The benchmark library is not identified as a release build.")
    if context.get("cpu_scaling_enabled") is True:
        messages.append("CPU frequency scaling was enabled and may increase timing variance.")
    load = context.get("load_avg")
    cpus = finite_number(context.get("num_cpus"))
    if isinstance(load, list) and load and cpus and finite_number(load[0]) is not None and float(load[0]) >= cpus:
        messages.append("The one-minute load average was at least the logical CPU count.")
    return messages


STYLE = """
:root{color-scheme:light;--bg:#f8fafc;--panel:#fff;--text:#172033;--muted:#64748b;--line:#dbe3ee;--accent:#2563eb;--good:#047857;--bad:#b91c1c;--warn:#92400e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px}h1{font-size:2rem;margin:0}h2{font-size:1.35rem;margin:38px 0 14px}h3{font-size:1.05rem;margin:0}p{color:var(--muted);margin:5px 0 14px}.subtitle{font-size:1rem}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}.card,.chart-card,.notice,.table-wrap{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 1px 2px #0f172a0a}.card{padding:14px 16px}.card span{display:block;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}.card strong{display:block;margin-top:3px;font-size:1rem}.notice{border-left:5px solid #f59e0b;padding:12px 18px;margin:14px 0}.notice strong{color:var(--warn)}.notice ul{margin:6px 0;padding-left:22px}.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}.chart-card{padding:16px;min-width:0}.chart-card.empty{min-height:150px}.chart{display:block;width:100%;height:auto}.grid{stroke:#e2e8f0;stroke-width:1}.axis-label,.legend{fill:#64748b;font-size:12px}.axis-title{fill:#334155;font-size:12px;font-weight:600}.series{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.point{stroke:#fff;stroke-width:2}.bar{opacity:.9}.table-wrap{overflow:auto;margin:14px 0}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #e8edf4;text-align:right}th{position:sticky;top:0;background:#f1f5f9;color:#334155;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em;cursor:pointer}th:first-child,th:nth-child(2),td:first-child,td:nth-child(2){text-align:left}tbody tr:hover{background:#f8fafc}.delta.positive{color:var(--good);font-weight:700}.delta.negative{color:var(--bad);font-weight:700}.filter{display:flex;align-items:center;gap:10px;margin:12px 0;color:#475569;font-weight:600}.filter input{min-width:300px;max-width:100%;padding:9px 12px;border:1px solid #cbd5e1;border-radius:8px;background:white}footer{margin-top:38px;color:var(--muted);font-size:.85rem}@media(max-width:600px){main{padding:20px 12px}.chart-grid{grid-template-columns:1fr}.filter{display:block}.filter input{display:block;margin-top:6px;width:100%;min-width:0}}@media print{body{background:white}main{max-width:none;padding:0}.chart-card,.card,.table-wrap,.notice{box-shadow:none;break-inside:avoid}.filter{display:none}th{position:static}}
"""


SCRIPT = """
document.querySelectorAll('[data-table-filter]').forEach(input=>{input.addEventListener('input',()=>{const table=document.getElementById(input.dataset.tableFilter);const query=input.value.trim().toLowerCase();table.querySelectorAll('tbody tr').forEach(row=>{row.hidden=!row.dataset.search.includes(query)})})});
document.querySelectorAll('th[data-sortable]').forEach(header=>{const activate=()=>{const table=header.closest('table');const body=table.tBodies[0];const index=[...header.parentElement.children].indexOf(header);const ascending=header.dataset.direction!=='asc';[...header.parentElement.children].forEach(item=>delete item.dataset.direction);header.dataset.direction=ascending?'asc':'desc';const rows=[...body.rows];rows.sort((a,b)=>{const av=a.cells[index].dataset.sort||'';const bv=b.cells[index].dataset.sort||'';const an=Number(av),bn=Number(bv);const value=av!==''&&bv!==''&&Number.isFinite(an)&&Number.isFinite(bn)?an-bn:av.localeCompare(bv);return ascending?value:-value});rows.forEach(row=>body.appendChild(row))};header.addEventListener('click',activate);header.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();activate()}})});
"""


def render_report(title: str, context: Mapping[str, Any], results: Sequence[Result], parser_warnings: Sequence[str]) -> str:
    search_results = [result for result in results if result.operation == "search"] or list(results)
    canonical = {
        "count": preferred_value(search_results, "count", 10_000),
        "dimension": preferred_value(search_results, "dimension", 128),
        "top_k": preferred_value(search_results, "top_k", 10),
        "query_count": preferred_value([result for result in results if result.operation == "batch_search"], "query_count", 8),
    }
    warnings = quality_warnings(context, results, parser_warnings)
    warning_html = ""
    if warnings:
        warning_html = '<section class="notice"><strong>Measurement notes</strong><ul>' + "".join(f"<li>{escape(message)}</li>" for message in warnings) + "</ul></section>"
    canonical_description = ", ".join(f"{key.replace('_', ' ')}={fmt_param(value)}" for key, value in canonical.items())
    overview = canonical_results(results, canonical)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{STYLE}</style></head>
<body><main><header><h1>{escape(title)}</h1><p class="subtitle">Human-readable summary of one Google Benchmark result file.</p></header>
{context_cards(context, results)}{warning_html}
<h2>Canonical workload</h2><p>{escape(canonical_description)}. Wall time is the primary latency; reported items/s may use Google Benchmark's CPU clock.</p>
{results_table(overview, "canonical-results", False)}
<h2>Search scaling</h2><p>Each chart changes one workload dimension while holding the others at the canonical values. Lower is better.</p>
{make_search_charts(results, canonical)}
<h2>LSH recall and latency</h2><p>Latency is measured per approximate cosine query. Recall is measured against exact FlatIndex top-k results; higher is better.</p>
{make_lsh_section(results)}
<h2>Batch search versus repeated searches</h2><p>Wall time is divided by the number of queries so the two APIs are directly comparable.</p>
{make_batch_section(results, canonical)}
<h2>Insertion and persistence</h2>
{make_write_charts(results)}
<h2>Complete results</h2><p>Click a column heading to sort. Effective bytes/s is a modeled scan rate, not measured disk or network traffic.</p>
{results_table(results, "all-results", True)}
<footer>Generated from Google Benchmark JSON by benchmarks/generate_report.py. The report is self-contained and makes no network requests.</footer>
</main><script>{SCRIPT}</script></body></html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Google Benchmark JSON input")
    parser.add_argument("-o", "--output", type=Path, help="HTML output (default: <input-stem>-report.html)")
    parser.add_argument("--title", default="Vector DB benchmark report", help="report heading")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.input.with_name(f"{args.input.stem}-report.html")
    try:
        context, results, warnings = load_results(args.input)
        report = render_report(args.title, context, results, warnings)
        output.write_text(report, encoding="utf-8")
    except (OSError, ReportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {output} ({len(results)} benchmark cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
