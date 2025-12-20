#!/usr/bin/env python3
"""
Generate Grafana dashboard JSON dynamically based on cluster topology.

This script reads topology.json and generates a dashboard with panels for each cluster:
- Cluster convergence status (stat panel)
- Accuracy vs round (xy chart)
- Per-node accuracy within cluster (time series)
"""

import json
import re
from pathlib import Path
from typing import Any

# Time window for last_over_time() to preserve metrics after training stops.
RETENTION_WINDOW = "1h"


def wrap_metric_with_retention(query: str) -> str:
    """Wrap metric selectors with last_over_time() to preserve data after scraping stops.

    Prometheus marks metrics as stale ~5 minutes after nodes stop sending them.
    Using last_over_time(metric[1h]) ensures stat/gauge panels show the last
    known value even after training completes.
    """
    # Pattern to match metric selectors like: metric_name or metric_name{labels}
    # This handles nested aggregations by wrapping the innermost metric.
    pattern = r'(fl_\w+)(\{[^}]*\})?(?!\[)'

    def replacer(match: re.Match) -> str:
        metric = match.group(1)
        labels = match.group(2) or ""
        return f"last_over_time({metric}{labels}[{RETENTION_WINDOW}])"

    return re.sub(pattern, replacer, query)


def create_stat_panel(
    title: str,
    query: str,
    grid_pos: dict[str, int],
    thresholds: list[dict] | None = None,
    unit: str = "none",
    use_retention: bool = True,
) -> dict[str, Any]:
    """Create a stat panel configuration.

    Args:
        use_retention: If True, wrap query with last_over_time() to preserve
            data after training stops.
    """
    if thresholds is None:
        thresholds = [
            {"color": "red", "value": None},
            {"color": "green", "value": 1},
        ]

    final_query = wrap_metric_with_retention(query) if use_retention else query

    return {
        "id": None,
        "type": "stat",
        "title": title,
        "gridPos": grid_pos,
        "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": {"mode": "absolute", "steps": thresholds},
                "color": {"mode": "thresholds"},
                "mappings": [],
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"values": False, "calcs": ["lastNotNull"], "fields": ""},
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": "none",
            "justifyMode": "auto",
        },
        "targets": [
            {
                "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
                "expr": final_query,
                "refId": "A",
                "legendFormat": "__auto",
            }
        ],
    }


def create_timeseries_panel(
    title: str,
    targets: list[dict],
    grid_pos: dict[str, int],
    unit: str = "percent",
    legend_mode: str = "list",
    axis_label: str = "",
) -> dict[str, Any]:
    """Create a time series panel configuration."""
    return {
        "id": None,
        "type": "timeseries",
        "title": title,
        "gridPos": grid_pos,
        "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": "line",
                    "lineInterpolation": "smooth",
                    "lineWidth": 2,
                    "fillOpacity": 10,
                    "gradientMode": "none",
                    "spanNulls": True,
                    "showPoints": "auto",
                    "pointSize": 5,
                    "stacking": {"mode": "none", "group": "A"},
                    "axisPlacement": "auto",
                    "axisLabel": axis_label,
                    "scaleDistribution": {"type": "linear"},
                },
                "color": {"mode": "palette-classic"},
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
                "mappings": [],
            },
            "overrides": [],
        },
        "options": {
            "tooltip": {"mode": "multi", "sort": "desc"},
            "legend": {
                "showLegend": True,
                "displayMode": legend_mode,
                "placement": "bottom",
                "calcs": ["lastNotNull", "max"],
            },
        },
        "targets": targets,
    }


def create_accuracy_vs_round_panel(cluster_id: int, grid_pos: dict[str, int]) -> dict[str, Any]:
    """Create an XY chart panel for accuracy vs round."""
    return {
        "id": None,
        "type": "barchart",
        "title": f"Cluster {cluster_id} - Accuracy vs Round",
        "gridPos": grid_pos,
        "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
        "fieldConfig": {
            "defaults": {
                "unit": "percent",
                "color": {"mode": "palette-classic"},
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
            },
            "overrides": [
                {
                    "matcher": {"id": "byName", "options": "round"},
                    "properties": [{"id": "unit", "value": "none"}],
                }
            ],
        },
        "options": {
            "xField": "round",
            "orientation": "vertical",
            "barWidth": 0.9,
            "groupWidth": 0.7,
            "barRadius": 0,
            "showValue": "auto",
            "stacking": "none",
            "tooltip": {"mode": "multi"},
            "legend": {
                "showLegend": True,
                "displayMode": "list",
                "placement": "bottom",
            },
        },
        "targets": [
            {
                "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
                "expr": (
                    "avg by (round) "
                    f'(last_over_time(fl_accuracy_by_round{{clique_id="{cluster_id}", dataset="train"}}[{RETENTION_WINDOW}])) * 100'
                ),
                "refId": "Training",
                "legendFormat": "Training",
                "format": "table",
                "instant": True,
            },
            {
                "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
                "expr": (
                    "avg by (round) "
                    f'(last_over_time(fl_accuracy_by_round{{clique_id="{cluster_id}", dataset="validation"}}[{RETENTION_WINDOW}])) * 100'
                ),
                "refId": "Validation",
                "legendFormat": "Validation",
                "format": "table",
                "instant": True,
            },
            {
                "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
                "expr": (
                    "avg by (round) "
                    f'(last_over_time(fl_accuracy_by_round{{clique_id="{cluster_id}", dataset="test"}}[{RETENTION_WINDOW}])) * 100'
                ),
                "refId": "Test",
                "legendFormat": "Test",
                "format": "table",
                "instant": True,
            },
        ],
        "transformations": [
            {
                "id": "labelsToFields",
                "options": {"mode": "columns", "keepLabels": ["round"]},
            },
            {
                "id": "joinByField",
                "options": {"byField": "round", "mode": "outer"},
            },
            {
                "id": "organize",
                "options": {
                    "excludeByName": {
                        "Time": True,
                        "Time 1": True,
                        "Time 2": True,
                        "Time 3": True,
                        "round 1": True,
                        "round 2": True,
                        "round 3": True,
                        "__name__": True,
                    }
                },
            },
            {
                "id": "renameByRegex",
                "options": {"regex": "Value #(.*)", "renamePattern": "$1"},
            },
            {
                "id": "convertFieldType",
                "options": {
                    "conversions": [
                        {"destinationType": "number", "targetField": "round"},
                        {"destinationType": "number", "targetField": "Training"},
                        {"destinationType": "number", "targetField": "Validation"},
                        {"destinationType": "number", "targetField": "Test"},
                    ]
                },
            },
            {
                "id": "sortBy",
                "options": {
                    "fields": {},
                    "sort": [{"desc": True, "field": "round"}],
                },
            },
            {
                "id": "limit",
                "options": {"count": 20},
            },
            {
                "id": "sortBy",
                "options": {
                    "fields": {},
                    "sort": [{"desc": False, "field": "round"}],
                },
            },
        ],
    }


def create_row_panel(title: str, grid_pos: dict[str, int]) -> dict[str, Any]:
    """Create a row panel for organizing sections."""
    return {
        "id": None,
        "type": "row",
        "title": title,
        "gridPos": grid_pos,
        "collapsed": False,
        "panels": [],
    }


def create_gauge_panel(
    title: str,
    query: str,
    grid_pos: dict[str, int],
    min_val: float = 0,
    max_val: float = 100,
    unit: str = "percent",
    use_retention: bool = True,
) -> dict[str, Any]:
    """Create a gauge panel configuration.

    Args:
        use_retention: If True, wrap query with last_over_time() to preserve
            data after training stops.
    """
    final_query = wrap_metric_with_retention(query) if use_retention else query

    return {
        "id": None,
        "type": "gauge",
        "title": title,
        "gridPos": grid_pos,
        "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "min": min_val,
                "max": max_val,
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "red", "value": None},
                        {"color": "yellow", "value": 50},
                        {"color": "green", "value": 80},
                    ],
                },
                "color": {"mode": "thresholds"},
                "mappings": [],
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"values": False, "calcs": ["lastNotNull"], "fields": ""},
            "showThresholdLabels": False,
            "showThresholdMarkers": True,
            "orientation": "auto",
        },
        "targets": [
            {
                "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
                "expr": final_query,
                "refId": "A",
            }
        ],
    }


def generate_cluster_panels(
    cluster_id: int, nodes: list[str], y_offset: int
) -> tuple[list[dict], int]:
    """Generate all panels for a single cluster."""
    panels = []
    current_y = y_offset

    # Row header.
    panels.append(
        create_row_panel(f"Cluster {cluster_id}", {"x": 0, "y": current_y, "w": 24, "h": 1})
    )
    current_y += 1

    # Stat panels row.
    panels.append(
        create_stat_panel(
            title="Converged",
            query=f'min(fl_cluster_converged{{clique_id="{cluster_id}"}})',
            grid_pos={"x": 0, "y": current_y, "w": 4, "h": 4},
            thresholds=[{"color": "red", "value": None}, {"color": "green", "value": 1}],
        )
    )
    panels.append(
        create_stat_panel(
            title="Current Round",
            query=f'max(fl_current_round{{clique_id="{cluster_id}"}})',
            grid_pos={"x": 4, "y": current_y, "w": 4, "h": 4},
            thresholds=[{"color": "blue", "value": None}],
        )
    )
    panels.append(
        create_stat_panel(
            title="Avg Test Accuracy",
            query=f'avg(fl_accuracy{{clique_id="{cluster_id}", dataset="test"}}) * 100',
            grid_pos={"x": 8, "y": current_y, "w": 4, "h": 4},
            thresholds=[
                {"color": "red", "value": None},
                {"color": "yellow", "value": 50},
                {"color": "green", "value": 80},
            ],
            unit="percent",
        )
    )
    panels.append(
        create_stat_panel(
            title="Convergence Streak",
            query=f'avg(fl_convergence_streak{{clique_id="{cluster_id}"}})',
            grid_pos={"x": 12, "y": current_y, "w": 4, "h": 4},
            thresholds=[{"color": "yellow", "value": None}, {"color": "green", "value": 3}],
        )
    )
    panels.append(
        create_stat_panel(
            title="Active Nodes",
            query=f'count(fl_current_round{{clique_id="{cluster_id}"}})',
            grid_pos={"x": 16, "y": current_y, "w": 4, "h": 4},
            thresholds=[{"color": "blue", "value": None}],
        )
    )
    current_y += 4

    # Accuracy vs round (Train/Val/Test).
    panels.append(
        create_accuracy_vs_round_panel(
            cluster_id=cluster_id,
            grid_pos={"x": 0, "y": current_y, "w": 12, "h": 8},
        )
    )

    # Convergence metrics over time.
    panels.append(
        create_timeseries_panel(
            title=f"Cluster {cluster_id} - Convergence Metrics",
            targets=[
                {
                    "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
                    "expr": f'avg(fl_convergence_streak{{clique_id="{cluster_id}"}})',
                    "refId": "Streak",
                    "legendFormat": "Convergence Streak",
                },
                {
                    "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
                    "expr": f'avg(fl_delta_norm{{clique_id="{cluster_id}"}}) * 100',
                    "refId": "Delta",
                    "legendFormat": "Delta Norm (×100)",
                },
            ],
            grid_pos={"x": 12, "y": current_y, "w": 12, "h": 8},
            unit="none",
            axis_label="Value",
        )
    )
    current_y += 8

    # Per-node test accuracy.
    node_targets = [
        {
            "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
            "expr": f'fl_accuracy{{clique_id="{cluster_id}", node_id="{node}", dataset="test"}} * 100',
            "refId": node,
            "legendFormat": node,
        }
        for node in nodes
    ]
    panels.append(
        create_timeseries_panel(
            title=f"Cluster {cluster_id} - Per-Node Test Accuracy",
            targets=node_targets,
            grid_pos={"x": 0, "y": current_y, "w": 12, "h": 8},
            unit="percent",
        )
    )

    # Per-node delta norm.
    delta_targets = [
        {
            "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
            "expr": f'fl_delta_norm{{clique_id="{cluster_id}", node_id="{node}"}}',
            "refId": node,
            "legendFormat": node,
        }
        for node in nodes
    ]
    panels.append(
        create_timeseries_panel(
            title=f"Cluster {cluster_id} - Per-Node Delta Norm",
            targets=delta_targets,
            grid_pos={"x": 12, "y": current_y, "w": 12, "h": 8},
            unit="none",
        )
    )
    current_y += 8

    return panels, current_y


def generate_overview_panels(num_clusters: int) -> tuple[list[dict], int]:
    """Generate overview panels showing all clusters at a glance."""
    panels = []
    current_y = 0

    panels.append(
        create_row_panel("Overview - All Clusters", {"x": 0, "y": current_y, "w": 24, "h": 1})
    )
    current_y += 1

    # Global stats row: Total Clusters, Global Round, All Converged.
    panels.append(
        create_stat_panel(
            title="Total Clusters",
            query=f'count(count by (clique_id) (fl_current_round))',
            grid_pos={"x": 0, "y": current_y, "w": 4, "h": 3},
            thresholds=[{"color": "blue", "value": None}],
        )
    )
    panels.append(
        create_stat_panel(
            title="Global Round",
            query='max(fl_current_round)',
            grid_pos={"x": 4, "y": current_y, "w": 4, "h": 3},
            thresholds=[{"color": "purple", "value": None}],
        )
    )
    panels.append(
        create_stat_panel(
            title="All Clusters Converged",
            query='min(fl_cluster_converged)',
            grid_pos={"x": 8, "y": current_y, "w": 4, "h": 3},
            thresholds=[{"color": "red", "value": None}, {"color": "green", "value": 1}],
        )
    )
    panels.append(
        create_stat_panel(
            title="Total Active Nodes",
            query='count(fl_current_round)',
            grid_pos={"x": 12, "y": current_y, "w": 4, "h": 3},
            thresholds=[{"color": "blue", "value": None}],
        )
    )
    panels.append(
        create_stat_panel(
            title="Avg Global Accuracy",
            query='avg(fl_accuracy{dataset="test"}) * 100',
            grid_pos={"x": 16, "y": current_y, "w": 4, "h": 3},
            thresholds=[
                {"color": "red", "value": None},
                {"color": "yellow", "value": 50},
                {"color": "green", "value": 80},
            ],
            unit="percent",
        )
    )
    panels.append(
        create_stat_panel(
            title="Clusters Converged",
            query='sum(min by (clique_id) (fl_cluster_converged))',
            grid_pos={"x": 20, "y": current_y, "w": 4, "h": 3},
            thresholds=[{"color": "yellow", "value": None}, {"color": "green", "value": num_clusters}],
        )
    )
    current_y += 3

    # Rounds over time for all clusters.
    round_targets = [
        {
            "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
            "expr": f'max(fl_current_round{{clique_id="{i}"}})',
            "refId": f"Cluster{i}",
            "legendFormat": f"Cluster {i}",
        }
        for i in range(num_clusters)
    ]
    round_targets.append(
        {
            "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
            "expr": 'max(fl_current_round)',
            "refId": "Global",
            "legendFormat": "Global Max",
        }
    )
    panels.append(
        create_timeseries_panel(
            title="Training Rounds Over Time",
            targets=round_targets,
            grid_pos={"x": 0, "y": current_y, "w": 12, "h": 6},
            unit="none",
            legend_mode="table",
            axis_label="Round",
        )
    )

    # All clusters convergence status over time.
    convergence_targets = [
        {
            "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
            "expr": f'min(fl_cluster_converged{{clique_id="{i}"}})',
            "refId": f"Cluster{i}",
            "legendFormat": f"Cluster {i}",
        }
        for i in range(num_clusters)
    ]
    panels.append(
        create_timeseries_panel(
            title="Cluster Convergence Status Over Time",
            targets=convergence_targets,
            grid_pos={"x": 12, "y": current_y, "w": 12, "h": 6},
            unit="none",
            legend_mode="table",
            axis_label="Converged (0/1)",
        )
    )
    current_y += 6

    # All clusters comparison.
    comparison_targets = [
        {
            "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
            "expr": f'avg(fl_accuracy{{clique_id="{i}", dataset="test"}}) * 100',
            "refId": f"Cluster{i}",
            "legendFormat": f"Cluster {i}",
        }
        for i in range(num_clusters)
    ]
    panels.append(
        create_timeseries_panel(
            title="All Clusters - Test Accuracy Comparison",
            targets=comparison_targets,
            grid_pos={"x": 0, "y": current_y, "w": 24, "h": 6},
            unit="percent",
            legend_mode="table",
        )
    )
    current_y += 6

    return panels, current_y


def generate_timing_panels(num_clusters: int, y_offset: int) -> tuple[list[dict], int]:
    """Generate timing metrics panels."""
    panels = []
    current_y = y_offset

    panels.append(
        create_row_panel("Timing Metrics", {"x": 0, "y": current_y, "w": 24, "h": 1})
    )
    current_y += 1

    for i in range(num_clusters):
        panels.append(
            create_timeseries_panel(
                title=f"Cluster {i} - Round Timing",
                targets=[
                    {
                        "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
                        "expr": f'avg(rate(fl_local_training_seconds_sum{{clique_id="{i}"}}[1m]) / rate(fl_local_training_seconds_count{{clique_id="{i}"}}[1m]))',
                        "refId": "Training",
                        "legendFormat": "Avg Training Time",
                    },
                    {
                        "datasource": {"type": "prometheus", "uid": "DS_PROMETHEUS"},
                        "expr": f'avg(rate(fl_aggregation_seconds_sum{{clique_id="{i}"}}[1m]) / rate(fl_aggregation_seconds_count{{clique_id="{i}"}}[1m]))',
                        "refId": "Aggregation",
                        "legendFormat": "Avg Aggregation Time",
                    },
                ],
                grid_pos={"x": (i % 2) * 12, "y": current_y + (i // 2) * 6, "w": 12, "h": 6},
                unit="s",
            )
        )

    current_y += ((num_clusters + 1) // 2) * 6

    return panels, current_y


def generate_dashboard(topology: dict) -> dict:
    """Generate the complete Grafana dashboard configuration."""
    num_clusters = topology.get("num_cliques", 1)
    cliques = topology.get("cliques", [])

    panels = []
    current_y = 0

    overview_panels, current_y = generate_overview_panels(num_clusters)
    panels.extend(overview_panels)

    for cluster_id in range(num_clusters):
        nodes = cliques[cluster_id] if cluster_id < len(cliques) else []
        cluster_panels, current_y = generate_cluster_panels(cluster_id, nodes, current_y)
        panels.extend(cluster_panels)

    timing_panels, current_y = generate_timing_panels(num_clusters, current_y)
    panels.extend(timing_panels)

    for i, panel in enumerate(panels):
        panel["id"] = i + 1

    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": panels,
        "refresh": "5s",
        "schemaVersion": 39,
        "tags": ["federated-learning", "secure-aggregation", "auto-generated"],
        "templating": {"list": []},
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": "Federated Learning - Cluster Metrics",
        "uid": "fl-cluster-metrics",
        "version": 1,
        "weekStart": "",
    }


def main() -> None:
    """Main entry point for dashboard generation."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if Path("/app/config/topology.json").exists():
        topology_path = Path("/app/config/topology.json")
        output_path = Path("/app/output/fl_metrics.json")
    else:
        topology_path = project_root / "config" / "topology.json"
        output_path = project_root / "docker" / "grafana" / "dashboards" / "fl_metrics.json"

    if not topology_path.exists():
        print(f"Error: Topology file not found at {topology_path}")
        raise SystemExit(1)

    with open(topology_path) as f:
        topology = json.load(f)

    num_clusters = topology.get("num_cliques", 1)
    cliques = topology.get("cliques", [])
    total_nodes = sum(len(c) for c in cliques)

    print(f"Generating Grafana dashboard for {num_clusters} clusters ({total_nodes} nodes)...")
    for i, nodes in enumerate(cliques):
        print(f"  Cluster {i}: {', '.join(nodes)}")

    dashboard = generate_dashboard(topology)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"Dashboard written to {output_path}")
    print(f"Total panels generated: {len(dashboard['panels'])}")


if __name__ == "__main__":
    main()
