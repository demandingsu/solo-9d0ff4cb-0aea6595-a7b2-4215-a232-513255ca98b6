"""Road-network edge matching (接边): candidates, mutual nearest, replacement."""

from __future__ import annotations

import json

import pytest

from app import roadnet
from tests.conftest import mask_bytes, png_bytes  # noqa: F401  (re-exported helpers)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _publish(client) -> str:
    response = client.post("/publish", follow_redirects=False)
    assert response.status_code == 303
    return client.get("/versions").json()[0]["id"]


def _configure(client, version_id: str, tolerance: float | str = 8):
    return client.post(
        "/roadnet/config",
        data={"version_id": version_id, "tolerance": str(tolerance)},
        follow_redirects=False,
    )


def _draft(client, tile_id: str, revision: int, obj):
    payload = obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")
    files = {
        "file": ("draft.json", payload, "application/json"),
        "tile_id": (None, tile_id),
        "revision": (None, str(revision)),
    }
    return client.post("/roadnet/drafts", files=files, follow_redirects=False)


def _net(client) -> dict:
    response = client.get("/roadnet/download")
    assert response.status_code == 200
    return response.json()


def _by_kind(net: dict) -> dict[str, list]:
    out: dict[str, list] = {"polyline": [], "join": [], "pending_endpoint": []}
    for feature in net["features"]:
        out[feature["properties"]["kind"]].append(feature)
    return out


def _line(points, line_id="l1", join=("start", "end")):
    return {"id": line_id, "points": [list(p) for p in points], "join": list(join)}


@pytest.fixture()
def frame(client, upload, canvas):
    """Published version: A[0,0] B[80,0] C[160,0], each 100x100, canvas 300x120."""
    assert canvas(300, 120).status_code == 303
    upload(tile_id="A", revision=1, x=0, y=0, width=100, height=100)
    upload(tile_id="B", revision=1, x=80, y=0, width=100, height=100)
    upload(tile_id="C", revision=1, x=160, y=0, width=100, height=100)
    version_id = _publish(client)
    assert "err=" not in _configure(client, version_id, 8).headers["location"]
    return version_id


@pytest.fixture()
def ab_drafts(frame, client):
    """A.a1 end (95,50); B.b1 start (94,50) + B.b2 end (165,50); C.c1 (163,50)."""
    assert "err=" not in _draft(client, "A", 1, {
        "polylines": [_line([[50, 50], [95, 50]], "a1", join=["end"])],
    }).headers["location"]
    assert "err=" not in _draft(client, "B", 1, {
        "polylines": [
            _line([[14, 50], [40, 50]], "b1", join=["start"]),
            _line([[40, 70], [85, 50]], "b2", join=["end"]),
        ],
    }).headers["location"]
    assert "err=" not in _draft(client, "C", 1, {
        "polylines": [_line([[3, 50], [50, 50]], "c1", join=["start"])],
    }).headers["location"]
    return frame


# --------------------------------------------------------------------------- #
# Pure matching logic
# --------------------------------------------------------------------------- #

PLACEMENTS = {
    "A": roadnet.Placement("A", 1, 0, 0, 100, 100),
    "B": roadnet.Placement("B", 1, 80, 0, 100, 100),
}


def _ep(eid, tile, x, y):
    return roadnet.Endpoint(
        id=eid, draft_id=eid, tile_id=tile, line_id="l", end_kind="end",
        tx=x, ty=y, fx=x, fy=y,
    )


def test_overlap_rect_inclusive_bounds():
    a = roadnet.Placement("A", 1, 0, 0, 100, 100)
    b = roadnet.Placement("B", 1, 80, 10, 100, 100)
    assert roadnet.overlap_rect(a, b) == (80, 10, 100, 100)
    far = roadnet.Placement("C", 1, 500, 0, 10, 10)
    assert roadnet.overlap_rect(a, far) is None
    touching = roadnet.Placement("D", 1, 100, 0, 10, 10)
    assert roadnet.overlap_rect(a, touching) == (100, 0, 100, 10)


def test_mutual_nearest_connects_only_when_reciprocal():
    eps = [_ep(1, "A", 95, 50), _ep(2, "B", 94, 50), _ep(3, "B", 85, 50)]
    edges = roadnet.compute_edges_for([1, 2, 3], eps, PLACEMENTS, 8.0)
    # 1<->2 are mutual nearest (d=1); 3's nearest is 1 but 1's nearest is 2.
    assert set(edges) == {(1, 2)}
    assert edges[(1, 2)] == pytest.approx(1.0)


def test_equidistant_candidates_are_not_unique():
    # Endpoint 1 sees endpoints 2 and 3 at exactly the same nearest distance.
    eps = [_ep(1, "A", 90, 50), _ep(2, "B", 93, 46), _ep(3, "B", 93, 54)]
    assert roadnet.compute_edges_for([1, 2, 3], eps, PLACEMENTS, 8.0) == {}


def test_candidates_must_be_inside_overlap():
    # (106, 50) is within tolerance of (99, 50) but outside A∩B (x<=100).
    eps = [_ep(1, "A", 99, 50), _ep(2, "B", 106, 50)]
    assert roadnet.compute_edges_for([1, 2], eps, PLACEMENTS, 8.0) == {}


def test_same_tile_endpoints_are_never_candidates():
    eps = [_ep(1, "A", 95, 50), _ep(2, "A", 95, 52)]
    assert roadnet.compute_edges_for([1, 2], eps, PLACEMENTS, 8.0) == {}


def test_tolerance_boundary_is_inclusive():
    eps = [_ep(1, "A", 92, 50), _ep(2, "B", 100, 50)]
    assert set(roadnet.compute_edges_for([1, 2], eps, PLACEMENTS, 8.0)) == {(1, 2)}
    assert roadnet.compute_edges_for([1, 2], eps, PLACEMENTS, 7.9) == {}


def test_negative_frame_coordinates_match():
    placements = {
        "A": roadnet.Placement("A", 1, -50, 0, 100, 100),
        "B": roadnet.Placement("B", 1, 30, 0, 100, 100),
    }
    eps = [_ep(1, "A", 40, 50), _ep(2, "B", 41, 50)]
    assert set(roadnet.compute_edges_for([1, 2], eps, placements, 8.0)) == {(1, 2)}


# --------------------------------------------------------------------------- #
# Draft parsing / validation
# --------------------------------------------------------------------------- #


def test_parse_draft_defaults_and_marking():
    lines = roadnet.parse_draft(json.dumps({
        "polylines": [
            {"points": [[0, 0], [5, 5]]},
            {"id": "r2", "points": [[1, 1], [2, 2]], "join": ["end"]},
            {"id": "r3", "points": [[1, 1], [2, 2]], "join": []},
        ],
    }).encode())
    assert lines[0].line_id == "line-1"
    assert (lines[0].join_start, lines[0].join_end) == (True, True)
    assert (lines[1].join_start, lines[1].join_end) == (False, True)
    assert (lines[2].join_start, lines[2].join_end) == (False, False)


@pytest.mark.parametrize("payload", [
    b"{not json",
    b"[]",
    b"{}",
    b'{"polylines": []}',
    b'{"polylines": [{"points": [[1, 2]]}]}',          # too few points
    b'{"polylines": [{"points": [[1, 2], [3]]}]}',      # bad point shape
    b'{"polylines": [{"points": [[1, 2], [3, "x"]]}]}',  # non-numeric
    b'{"polylines": [{"points": [[1, 2], [3, true]]}]}',  # bool is not a number
    b'{"polylines": [{"points": [[1, 2], [3, 4]], "join": ["middle"]}]}',
    b'{"polylines": [{"id": "x", "points": [[0, 0], [1, 1]]},'
    b'              {"id": "x", "points": [[2, 2], [3, 3]]}]}',  # duplicate id
])
def test_parse_draft_rejects_bad_payloads(payload):
    with pytest.raises(roadnet.DraftError):
        roadnet.parse_draft(payload)


def test_check_points_within_tile():
    lines = roadnet.parse_draft(
        b'{"polylines": [{"points": [[0, 0], [100, 100]]}]}'
    )
    roadnet.check_points_within(lines, 100, 100)  # boundary inclusive
    with pytest.raises(roadnet.DraftError):
        roadnet.check_points_within(lines, 99, 100)


def test_validate_tolerance():
    assert roadnet.validate_tolerance("8") == 8.0
    assert roadnet.validate_tolerance("0.5") == 0.5
    for bad in ("0", "-1", "abc", "1e999", "200000"):
        with pytest.raises(roadnet.DraftError):
            roadnet.validate_tolerance(bad)


# --------------------------------------------------------------------------- #
# Web flow: configuration and upload validation
# --------------------------------------------------------------------------- #


def test_config_requires_existing_version(client, upload, canvas):
    canvas(10, 10)
    upload(tile_id="A", revision=1, width=2, height=2)
    _publish(client)
    assert "err=" in _configure(client, "no-such-version").headers["location"]
    assert "err=" in _configure(client, "", "abc").headers["location"]


def test_upload_requires_selected_version(client, upload, canvas):
    canvas(10, 10)
    upload(tile_id="A", revision=1, width=2, height=2)
    _publish(client)
    response = _draft(client, "A", 1, {"polylines": [_line([[0, 0], [1, 1]])]})
    assert "err=" in response.headers["location"]
    assert client.get("/roadnet/download").status_code == 409


def test_upload_validates_against_manifest(frame, client):
    base = {"polylines": [_line([[0, 0], [5, 5]])]}
    assert "err=" in _draft(client, "A", 2, base).headers["location"]   # wrong rev
    assert "err=" in _draft(client, "ZZ", 1, base).headers["location"]  # not in manifest
    out = {"polylines": [_line([[0, 0], [101, 5]])]}                    # x > width
    assert "err=" in _draft(client, "A", 1, out).headers["location"]
    assert "err=" in _draft(client, "A", 1, b"{broken").headers["location"]
    ok = _draft(client, "A", 1, base)
    assert "msg=" in ok.headers["location"]


# --------------------------------------------------------------------------- #
# Matching through the web app
# --------------------------------------------------------------------------- #


def test_frame_coordinate_conversion_in_download(ab_drafts, client):
    net = _net(client)
    polylines = {f["properties"]["line_id"]: f for f in _by_kind(net)["polyline"]}
    # Tile B sits at x=80: tile coords shift by the manifest placement.
    assert polylines["b1"]["geometry"]["coordinates"] == [[94.0, 50.0], [120.0, 50.0]]
    assert polylines["c1"]["geometry"]["coordinates"] == [[163.0, 50.0], [210.0, 50.0]]
    assert net["roadnet"]["coordinate_space"].startswith("frame pixels")


def test_mutual_nearest_edges_and_pending(ab_drafts, client):
    net = _net(client)
    parts = _by_kind(net)
    joins = parts["join"]
    assert len(joins) == 2
    pairs = {
        (j["properties"]["a"]["line_id"], j["properties"]["b"]["line_id"])
        for j in joins
    }
    assert pairs == {("a1", "b1"), ("b2", "c1")}
    distances = sorted(j["properties"]["distance"] for j in joins)
    assert distances == pytest.approx([1.0, 2.0])
    assert parts["pending_endpoint"] == []
    assert net["roadnet"]["counts"] == {
        "polylines": 4, "joins": 2, "pending_endpoints": 0,
    }


def test_equidistant_candidates_stay_unjoined(frame, client):
    # A endpoint (90,50); B endpoints (93,46) and (93,54): both at distance 5.
    _draft(client, "A", 1, {"polylines": [_line([[50, 50], [90, 50]], "a1", join=["end"])]})
    _draft(client, "B", 1, {"polylines": [
        _line([[13, 46], [40, 46]], "b1", join=["start"]),
        _line([[13, 54], [40, 54]], "b2", join=["start"]),
    ]})
    parts = _by_kind(_net(client))
    assert parts["join"] == []
    assert len(parts["pending_endpoint"]) == 3


def test_overlap_and_tolerance_enforced(frame, client):
    # Within tolerance but B's endpoint is outside the A∩B overlap (x>100).
    _draft(client, "A", 1, {"polylines": [_line([[50, 50], [99, 50]], "a1", join=["end"])]})
    _draft(client, "B", 1, {"polylines": [_line([[26, 50], [40, 50]], "b1", join=["start"])]})
    assert _by_kind(_net(client))["join"] == []

    # Exactly at tolerance: distance 8 == tol connects...
    _draft(client, "A", 1, {"polylines": [_line([[50, 50], [92, 50]], "a1", join=["end"])]})
    _draft(client, "B", 1, {"polylines": [_line([[20, 50], [40, 50]], "b1", join=["start"])]})
    assert len(_by_kind(_net(client))["join"]) == 1

    # ...while 8.5 px apart does not (replacing B's draft).
    _draft(client, "B", 1, {"polylines": [_line([[20.5, 50], [40, 50]], "b1", join=["start"])]})
    assert _by_kind(_net(client))["join"] == []


def test_same_tile_endpoints_not_joined(frame, client):
    _draft(client, "A", 1, {"polylines": [
        _line([[50, 50], [95, 50]], "a1", join=["end"]),
        _line([[95, 52], [60, 60]], "a2", join=["start"]),
    ]})
    parts = _by_kind(_net(client))
    assert parts["join"] == []
    assert len(parts["pending_endpoint"]) == 2


# --------------------------------------------------------------------------- #
# Draft replacement (换稿)
# --------------------------------------------------------------------------- #


def test_replace_draft_updates_its_edges(ab_drafts, client):
    # Move A's endpoint far away: the A-B edge disappears, B-C is untouched.
    _draft(client, "A", 1, {"polylines": [_line([[10, 10], [20, 20]], "a1", join=["end"])]})
    parts = _by_kind(_net(client))
    pairs = {
        (j["properties"]["a"]["line_id"], j["properties"]["b"]["line_id"])
        for j in parts["join"]
    }
    assert pairs == {("b2", "c1")}
    pending = {(p["properties"]["tile_id"], p["properties"]["line_id"])
               for p in parts["pending_endpoint"]}
    assert pending == {("A", "a1"), ("B", "b1")}


def test_replacement_can_break_edge_between_other_tiles(client, upload, canvas):
    # A[0,0] B[80,0] C[60,0] (C overlaps both A and B around x=95).
    canvas(300, 120)
    upload(tile_id="A", revision=1, x=0, y=0, width=100, height=100)
    upload(tile_id="B", revision=1, x=80, y=0, width=100, height=100)
    upload(tile_id="C", revision=1, x=60, y=0, width=100, height=100)
    version_id = _publish(client)
    _configure(client, version_id, 8)
    _draft(client, "A", 1, {"polylines": [_line([[50, 50], [95, 50]], "a1", join=["end"])]})
    _draft(client, "B", 1, {"polylines": [_line([[14, 50], [40, 50]], "b1", join=["start"])]})
    _draft(client, "C", 1, {"polylines": [_line([[10, 10], [20, 10]], "c1", join=["start"])]})
    pairs = {
        (j["properties"]["a"]["line_id"], j["properties"]["b"]["line_id"])
        for j in _by_kind(_net(client))["join"]
    }
    assert pairs == {("a1", "b1")}

    # Replace C: its new endpoint (95.2, 50) is closer to A's endpoint than B's.
    _draft(client, "C", 1, {"polylines": [_line([[35.2, 50], [60, 60]], "c1", join=["start"])]})
    parts = _by_kind(_net(client))
    pairs = {
        (j["properties"]["a"]["line_id"], j["properties"]["b"]["line_id"])
        for j in parts["join"]
    }
    assert pairs == {("a1", "c1")}
    pending = {(p["properties"]["tile_id"], p["properties"]["line_id"])
               for p in parts["pending_endpoint"]}
    assert pending == {("B", "b1")}


def test_failed_replace_keeps_previous_network(ab_drafts, client, monkeypatch):
    before = client.get("/roadnet/download").content

    # Invalid JSON never reaches the database.
    assert "err=" in _draft(client, "A", 1, b"{broken").headers["location"]
    assert client.get("/roadnet/download").content == before

    # A failure mid-update rolls back: draft and edges stay as they were.
    def boom(**kwargs):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(roadnet, "replace_draft", boom)
    response = _draft(client, "A", 1, {"polylines": [_line([[1, 1], [2, 2]], "a1")]})
    assert "err=" in response.headers["location"]
    assert client.get("/roadnet/download").content == before

    # The old draft is still the active one afterwards.
    monkeypatch.undo()
    parts = _by_kind(_net(client))
    assert len(parts["join"]) == 2
    assert any(p["properties"]["line_id"] == "a1"
               and p["geometry"]["coordinates"][-1] == [95.0, 50.0]
               for p in parts["polyline"])


def test_tolerance_change_recomputes_all_edges(ab_drafts, client):
    assert len(_by_kind(_net(client))["join"]) == 2
    _configure(client, ab_drafts, 0.5)  # below both join distances (1 and 2)
    parts = _by_kind(_net(client))
    assert parts["join"] == []
    assert len(parts["pending_endpoint"]) == 4
    _configure(client, ab_drafts, 8)
    assert len(_by_kind(_net(client))["join"]) == 2


def test_networks_are_isolated_per_version(ab_drafts, client, upload):
    first = ab_drafts
    assert len(_by_kind(_net(client))["join"]) == 2

    # Publish a second version and select it: it has no drafts of its own.
    upload(tile_id="A", revision=2, x=0, y=0, width=100, height=100,
           colour=(1, 2, 3))
    second = _publish(client)
    assert second != first
    _configure(client, second, 8)
    parts = _by_kind(_net(client))
    assert parts["polyline"] == [] and parts["join"] == []

    # Switching back restores the first version's network untouched.
    _configure(client, first, 8)
    assert len(_by_kind(_net(client))["join"]) == 2


# --------------------------------------------------------------------------- #
# Page and download surface
# --------------------------------------------------------------------------- #


def test_page_lists_pending_endpoints_and_edges(ab_drafts, client):
    # Replacing C's draft drops the b2-c1 edge; c9's endpoint stays pending.
    _draft(client, "C", 1, {"polylines": [_line([[3, 90], [50, 90]], "c9", join=["start"])]})
    page = client.get("/").text
    assert "接边（1 条）" in page
    assert "未接上的待接端点（2 个）" in page
    assert "c9" in page
    assert "下载画幅坐标路网 GeoJSON" in page


def test_download_requires_selection(client):
    assert client.get("/roadnet/download").status_code == 409


def test_download_empty_network_is_valid_geojson(frame, client):
    net = _net(client)
    assert net["type"] == "FeatureCollection"
    assert net["features"] == []
    assert net["roadnet"]["counts"] == {
        "polylines": 0, "joins": 0, "pending_endpoints": 0,
    }
