"""Tests for scripts/gridvision_lidar_probe.py — octree-key math only, no network.

The EPT walker's cell indexing is the part that silently breaks (a wrong index
returns 'no nodes' which reads as 'no coverage'). These tests pin it against a
fake hierarchy, including the two traps hit during the 2026-07-12 probe:
sub-hierarchy (-1) expansion and points spread across every tree level.
"""
import json
import sys
import unittest
from unittest import mock

sys.path.insert(0, "scripts")
import gridvision_lidar_probe as probe


def make_ept(cube=(0.0, 0.0, -10.0, 1024.0, 1024.0, 1014.0)):
    return {"bounds": list(cube)}


class CoveringNodesTest(unittest.TestCase):
    def test_ancestors_at_every_depth_are_returned(self):
        # point (100, 100): depth0 -> 0-0-0-*, depth1 cell 512 -> 1-0-0-*, depth2 cell 256 -> 2-0-0-*
        hier = {"0-0-0-0": 100, "1-0-0-0": 50, "2-0-0-1": 25}
        nodes = probe.covering_nodes("fake", make_ept(), hier, 100.0, 100.0)
        self.assertEqual(nodes, ["0-0-0-0", "1-0-0-0", "2-0-0-1"])

    def test_walk_stops_where_tree_ends(self):
        hier = {"0-0-0-0": 100}
        nodes = probe.covering_nodes("fake", make_ept(), hier, 100.0, 100.0)
        self.assertEqual(nodes, ["0-0-0-0"])

    def test_point_in_other_quadrant_skips_nonmatching_nodes(self):
        # point (900, 900) is in the (1,1) child at depth 1, not (0,0)
        hier = {"0-0-0-0": 100, "1-0-0-0": 50, "1-1-1-0": 60}
        nodes = probe.covering_nodes("fake", make_ept(), hier, 900.0, 900.0)
        self.assertEqual(nodes, ["0-0-0-0", "1-1-1-0"])

    def test_zero_point_nodes_are_excluded(self):
        hier = {"0-0-0-0": 100, "1-0-0-0": 0}
        nodes = probe.covering_nodes("fake", make_ept(), hier, 100.0, 100.0)
        self.assertEqual(nodes, ["0-0-0-0"])

    def test_subhierarchy_pointer_is_expanded(self):
        # depth-1 node is a -1 pointer; its sub-page holds the real count and a child
        hier = {"0-0-0-0": 100, "1-0-0-0": -1}
        sub_page = {"1-0-0-0": 40, "2-0-0-1": 20}

        def fake_fetch(url, *a, **kw):
            assert url.endswith("/ept-hierarchy/1-0-0-0.json"), url
            return json.dumps(sub_page).encode()

        with mock.patch.object(probe, "fetch", side_effect=fake_fetch):
            nodes = probe.covering_nodes("fake", make_ept(), hier, 100.0, 100.0)
        self.assertEqual(nodes, ["0-0-0-0", "1-0-0-0", "2-0-0-1"])

    def test_z_index_never_filters(self):
        # keys carry a z index the walker must ignore (points at any altitude count)
        hier = {"0-0-0-0": 10, "1-0-0-7": 5}
        nodes = probe.covering_nodes("fake", make_ept(), hier, 100.0, 100.0)
        self.assertIn("1-0-0-7", nodes)


if __name__ == "__main__":
    unittest.main()
