"""
test_tools.py - the parts of metrics.py that other components depend on. Run: pytest -q
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import metrics

EXAMPLE = json.loads((Path(__file__).parent / ".." / "supervisor" / "status.example.json").read_text())


def test_status_contract_parses_state_nodes_and_load_time():
    assert metrics.parse_status(EXAMPLE) == ("degraded", "2/4", "18.2")


def test_status_parse_tolerates_missing_fields():
    assert metrics.parse_status({}) == ("?", "", "")
    assert metrics.parse_status({"state": "down", "nodes_active": 0}) == ("down", "", "")


def test_short_name_keeps_ips_and_strips_local():
    assert metrics.short_name("pi-node-1.local") == "pi-node-1"
    assert metrics.short_name("192.168.50.11") == "192.168.50.11"


def test_csv_header_has_nineteen_columns_ending_in_the_new_ones():
    assert len(metrics.CSV_HEADER) == 19
    assert metrics.CSV_HEADER[-3:] == ["event", "cluster_nodes", "load_s"]
