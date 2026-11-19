"""Road-network edge matching (接边) across tiles of a published frame.

Cartographers trace road centrelines per tile; this module parses the
uploaded polyline drafts (tile-pixel coordinates), converts them to frame
coordinates using the selected published version's manifest, and computes
join edges between tiles:

* two endpoints are *candidates* only when they belong to different tiles,
  both lie inside the two tiles' overlap region, and their distance is
  within the fixed tolerance;
* an edge is created only when the two endpoints are each other's
  **unique** nearest candidate (an exact distance tie is not unique).

Replacing one tile's draft only re-examines the new/old pending endpoints
and the neighbouring endpoints within tolerance; every update commits the
draft and its edges in a single transaction, so a failed update leaves the
previous network downloadable.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

MAX_DRAFT_BYTES = 8 * 1024 * 1024
MAX_POLYLINES = 2000
MAX_POINTS_PER_LINE = 20000
MAX_TOTAL_POINTS = 200000
MAX_LINE_ID_LEN = 64
MAX_TOLERANCE = 100000.0
DEFAULT_TOLERANCE = 8.0

ENDS = ("start", "end")


class DraftError(ValueError):
    """Raised when a line draft or road-network setting is invalid."""


@dataclass(frozen=True)
class Polyline:
    line_id: str
    points: tuple[tuple[float, float], ...]
    join_start: bool
    join_end: bool


@dataclass(frozen=True)
class Placement:
    """Where one tile sits in the published frame (from the manifest)."""

    tile_id: str
    revision: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Endpoint:
    id: int
    draft_id: int
    tile_id: str
    line_id: str
    end_kind: str  # "start" | "end"
    tx: float
    ty: float
    fx: float
    fy: float


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #


def validate_tolerance(raw: str | float) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        raise DraftError("接边容差必须是数字。") from None
    if not math.isfinite(value) or value <= 0:
        raise DraftError("接边容差必须是正数。")
    if value > MAX_TOLERANCE:
        raise DraftError(f"接边容差不能超过 {MAX_TOLERANCE:g} 像素。")
    return value


def _check_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DraftError(f"{label}必须是数字。")
    result = float(value)
    if not math.isfinite(result):
        raise DraftError(f"{label}必须是有限数字。")
    return result


def parse_draft(payload: bytes) -> list[Polyline]:
    """Parse and validate the uploaded polyline JSON (tile-pixel coords).

    Expected shape::

        {"polylines": [{"id": "r1", "points": [[x, y], ...],
                        "join": ["start", "end"]}]}

    ``id`` defaults to ``line-<n>``; ``join`` defaults to both ends and
    marks which endpoints are pending joins (待接端点).
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise DraftError("线稿 JSON 必须是 UTF-8 编码。") from None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        raise DraftError("线稿文件不是合法的 JSON。") from None
    return parse_draft_obj(obj)


def parse_draft_obj(obj: object) -> list[Polyline]:
    if not isinstance(obj, dict):
        raise DraftError("线稿 JSON 顶层必须是对象。")
    lines = obj.get("polylines")
    if not isinstance(lines, list) or not lines:
        raise DraftError("线稿 JSON 需要非空的 polylines 数组。")
    if len(lines) > MAX_POLYLINES:
        raise DraftError(f"折线数量超过上限 {MAX_POLYLINES} 条。")

    result: list[Polyline] = []
    seen_ids: set[str] = set()
    total_points = 0
    for index, item in enumerate(lines):
        where = f"第 {index + 1} 条折线"
        if not isinstance(item, dict):
            raise DraftError(f"{where}必须是对象。")
        line_id = item.get("id") or f"line-{index + 1}"
        if not isinstance(line_id, str):
            raise DraftError(f"{where}的 id 必须是字符串。")
        line_id = line_id.strip()
        if not line_id:
            raise DraftError(f"{where}的 id 不能为空。")
        if len(line_id) > MAX_LINE_ID_LEN:
            raise DraftError(f"{where}的 id 不能超过 {MAX_LINE_ID_LEN} 个字符。")
        if line_id in seen_ids:
            raise DraftError(f"折线 id “{line_id}”重复。")
        seen_ids.add(line_id)

        points = item.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise DraftError(f"折线 {line_id} 至少需要 2 个点。")
        if len(points) > MAX_POINTS_PER_LINE:
            raise DraftError(
                f"折线 {line_id} 的点数超过上限 {MAX_POINTS_PER_LINE}。"
            )
        total_points += len(points)
        if total_points > MAX_TOTAL_POINTS:
            raise DraftError(f"总点数超过上限 {MAX_TOTAL_POINTS}。")
        parsed: list[tuple[float, float]] = []
        for order, point in enumerate(points):
            label = f"折线 {line_id} 第 {order + 1} 个点"
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise DraftError(f"{label}必须是 [x, y] 数组。")
            parsed.append(
                (
                    _check_number(point[0], f"{label}的 x"),
                    _check_number(point[1], f"{label}的 y"),
                )
            )

        join = item.get("join", ["start", "end"])
        if not isinstance(join, list) or any(j not in ENDS for j in join):
            raise DraftError(
                f"折线 {line_id} 的 join 只能是 [\"start\", \"end\"] 的子集。"
            )
        result.append(
            Polyline(
                line_id=line_id,
                points=tuple(parsed),
                join_start="start" in join,
                join_end="end" in join,
            )
        )
    return result


def check_points_within(
    polylines: Sequence[Polyline], width: int, height: int
) -> None:
    for line in polylines:
        for x, y in line.points:
            if not (0 <= x <= width and 0 <= y <= height):
                raise DraftError(
                    f"折线 {line.line_id} 的点 ({x:g}, {y:g}) 超出瓦片范围 "
                    f"0..{width} × 0..{height}。"
                )


# --------------------------------------------------------------------------- #
# Geometry and matching
# --------------------------------------------------------------------------- #


def overlap_rect(a: Placement, b: Placement) -> tuple[float, float, float, float] | None:
    """Frame-space overlap of two tiles, inclusive bounds (None if disjoint)."""
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.width, b.x + b.width)
    y1 = min(a.y + a.height, b.y + b.height)
    if x1 < x0 or y1 < y0:
        return None
    return (x0, y0, x1, y1)


def _inside(x: float, y: float, rect: tuple[float, float, float, float]) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


class _Grid:
    """Uniform grid (cell = tolerance) for tolerance-radius neighbour lookup."""

    def __init__(self, endpoints: Iterable[Endpoint], tolerance: float) -> None:
        self._tol = tolerance
        buckets: dict[tuple[int, int], list[Endpoint]] = defaultdict(list)
        for ep in endpoints:
            buckets[(math.floor(ep.fx / tolerance), math.floor(ep.fy / tolerance))].append(ep)
        self._buckets = buckets

    def near(self, ep: Endpoint) -> Iterator[Endpoint]:
        cx = math.floor(ep.fx / self._tol)
        cy = math.floor(ep.fy / self._tol)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from self._buckets.get((cx + dx, cy + dy), ())


def _dist2(a: Endpoint, b: Endpoint) -> float:
    return (a.fx - b.fx) ** 2 + (a.fy - b.fy) ** 2


def unique_nearest(
    ep: Endpoint,
    grid: _Grid,
    placements: dict[str, Placement],
    tolerance: float,
    _overlap_cache: dict | None = None,
) -> Endpoint | None:
    """The single nearest candidate of `ep`, or None when absent or tied.

    Candidates are endpoints of *other* tiles that lie, together with `ep`,
    inside the two tiles' overlap region and within the tolerance.
    """
    tol2 = tolerance * tolerance
    best: Endpoint | None = None
    best_d2 = 0.0
    tied = False
    overlaps = _overlap_cache if _overlap_cache is not None else {}
    for other in grid.near(ep):
        if other.tile_id == ep.tile_id:
            continue
        key = (ep.tile_id, other.tile_id)
        rect = overlaps.get(key, "unset")
        if rect == "unset":
            rect = overlap_rect(placements[ep.tile_id], placements[other.tile_id])
            overlaps[key] = rect
        if rect is None:
            continue
        if not (_inside(ep.fx, ep.fy, rect) and _inside(other.fx, other.fy, rect)):
            continue
        d2 = _dist2(ep, other)
        if d2 > tol2:
            continue
        if best is None or d2 < best_d2:
            best, best_d2, tied = other, d2, False
        elif d2 == best_d2:
            tied = True
    if best is None or tied:
        return None
    return best


def compute_edges_for(
    affected_ids: Iterable[int],
    endpoints: Sequence[Endpoint],
    placements: dict[str, Placement],
    tolerance: float,
) -> dict[tuple[int, int], float]:
    """Join edges touching `affected_ids`, as {(lo_id, hi_id): distance}.

    An edge exists exactly when both endpoints are each other's unique
    nearest candidate.  Nearest-neighbour results are cached so every
    endpoint is evaluated at most once per call.
    """
    by_id = {ep.id: ep for ep in endpoints}
    grid = _Grid(endpoints, tolerance)
    overlap_cache: dict = {}
    nearest_cache: dict[int, Endpoint | None] = {}

    def nearest(ep: Endpoint) -> Endpoint | None:
        if ep.id not in nearest_cache:
            nearest_cache[ep.id] = unique_nearest(
                ep, grid, placements, tolerance, overlap_cache
            )
        return nearest_cache[ep.id]

    found: dict[tuple[int, int], float] = {}
    for pid in affected_ids:
        ep = by_id.get(pid)
        if ep is None:
            continue
        other = nearest(ep)
        if other is None:
            continue
        back = nearest(other)
        if back is not None and back.id == ep.id:
            key = (min(ep.id, other.id), max(ep.id, other.id))
            found[key] = math.sqrt(_dist2(ep, other))
    return found


def neighbours_within(
    sources: Sequence[Endpoint],
    others: Sequence[Endpoint],
    tolerance: float,
) -> set[int]:
    """Ids of `others` (different tile) within tolerance of any source."""
    if not sources or not others:
        return set()
    grid = _Grid(others, tolerance)
    tol2 = tolerance * tolerance
    found: set[int] = set()
    for ep in sources:
        for other in grid.near(ep):
            if other.tile_id != ep.tile_id and _dist2(ep, other) <= tol2:
                found.add(other.id)
    return found


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

_ENDPOINT_COLS = "id, draft_id, tile_id, line_id, end_kind, tx, ty, fx, fy"


def _endpoint_from_row(row: sqlite3.Row) -> Endpoint:
    return Endpoint(
        id=row["id"],
        draft_id=row["draft_id"],
        tile_id=row["tile_id"],
        line_id=row["line_id"],
        end_kind=row["end_kind"],
        tx=row["tx"],
        ty=row["ty"],
        fx=row["fx"],
        fy=row["fy"],
    )


def load_endpoints(conn: sqlite3.Connection, version_id: str) -> list[Endpoint]:
    rows = conn.execute(
        f"SELECT {_ENDPOINT_COLS} FROM road_endpoints WHERE version_id = ? ORDER BY id",
        (version_id,),
    ).fetchall()
    return [_endpoint_from_row(row) for row in rows]


def load_active_drafts(conn: sqlite3.Connection, version_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM road_drafts WHERE version_id = ? AND status = 'active' "
        "ORDER BY tile_id",
        (version_id,),
    ).fetchall()


def _placements_from_drafts(drafts: Iterable[sqlite3.Row]) -> dict[str, Placement]:
    return {
        row["tile_id"]: Placement(
            tile_id=row["tile_id"],
            revision=row["revision"],
            x=row["tile_x"],
            y=row["tile_y"],
            width=row["tile_w"],
            height=row["tile_h"],
        )
        for row in drafts
    }


def _delete_edges_touching(
    conn: sqlite3.Connection, version_id: str, endpoint_ids: Sequence[int]
) -> None:
    for start in range(0, len(endpoint_ids), 400):
        chunk = list(endpoint_ids[start:start + 400])
        if not chunk:
            continue
        marks = ",".join("?" for _ in chunk)
        conn.execute(
            f"DELETE FROM road_edges WHERE version_id = ? AND "
            f"(endpoint_a IN ({marks}) OR endpoint_b IN ({marks}))",
            (version_id, *chunk, *chunk),
        )


def _insert_edges(
    conn: sqlite3.Connection,
    version_id: str,
    edges: dict[tuple[int, int], float],
    now: str,
) -> None:
    for (a, b), distance in sorted(edges.items()):
        conn.execute(
            "INSERT INTO road_edges(version_id, endpoint_a, endpoint_b, distance, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (version_id, a, b, distance, now),
        )


def replace_draft(
    conn: sqlite3.Connection,
    *,
    version_id: str,
    placement: Placement,
    polylines: Sequence[Polyline],
    payload_bytes: bytes,
    tolerance: float,
    now: str,
) -> dict[str, int]:
    """Replace one tile's draft and refresh only the affected join edges.

    Runs inside the caller's transaction: the new draft and the updated
    edges commit or roll back together.

    Only the new/old pending endpoints and neighbouring tiles' endpoints
    within the tolerance are re-examined; edges between untouched endpoints
    are left as they are.
    """
    drafts = load_active_drafts(conn, version_id)
    placements = _placements_from_drafts(drafts)
    placements[placement.tile_id] = placement
    endpoints = load_endpoints(conn, version_id)

    affected: set[int] = set()
    old = next((d for d in drafts if d["tile_id"] == placement.tile_id), None)
    if old is not None:
        old_eps = [ep for ep in endpoints if ep.draft_id == old["id"]]
        others = [ep for ep in endpoints if ep.draft_id != old["id"]]
        affected |= neighbours_within(old_eps, others, tolerance)
        _delete_edges_touching(conn, version_id, [ep.id for ep in old_eps])
        conn.execute(
            "DELETE FROM road_endpoints WHERE draft_id = ?", (old["id"],)
        )
        conn.execute(
            "UPDATE road_drafts SET status = 'superseded' WHERE id = ?",
            (old["id"],),
        )
        endpoints = others

    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    cursor = conn.execute(
        "INSERT INTO road_drafts(version_id, tile_id, revision, tile_x, tile_y, "
        "tile_w, tile_h, payload, payload_sha256, line_count, uploaded_at, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')",
        (
            version_id,
            placement.tile_id,
            placement.revision,
            placement.x,
            placement.y,
            placement.width,
            placement.height,
            payload_bytes.decode("utf-8"),
            payload_sha,
            len(polylines),
            now,
        ),
    )
    draft_id = int(cursor.lastrowid)

    new_eps: list[Endpoint] = []
    for line in polylines:
        for end_kind, point in (("start", line.points[0]), ("end", line.points[-1])):
            if end_kind == "start" and not line.join_start:
                continue
            if end_kind == "end" and not line.join_end:
                continue
            fx = placement.x + point[0]
            fy = placement.y + point[1]
            cursor = conn.execute(
                "INSERT INTO road_endpoints(draft_id, version_id, tile_id, "
                "line_id, end_kind, tx, ty, fx, fy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    draft_id,
                    version_id,
                    placement.tile_id,
                    line.line_id,
                    end_kind,
                    point[0],
                    point[1],
                    fx,
                    fy,
                ),
            )
            new_eps.append(
                Endpoint(
                    id=int(cursor.lastrowid),
                    draft_id=draft_id,
                    tile_id=placement.tile_id,
                    line_id=line.line_id,
                    end_kind=end_kind,
                    tx=point[0],
                    ty=point[1],
                    fx=fx,
                    fy=fy,
                )
            )

    affected |= {ep.id for ep in new_eps}
    affected |= neighbours_within(new_eps, endpoints, tolerance)
    _delete_edges_touching(conn, version_id, sorted(affected))

    all_eps = list(endpoints) + new_eps
    edges = compute_edges_for(affected, all_eps, placements, tolerance)
    _insert_edges(conn, version_id, edges, now)

    total_edges = conn.execute(
        "SELECT COUNT(*) FROM road_edges WHERE version_id = ?", (version_id,)
    ).fetchone()[0]
    return {
        "lines": len(polylines),
        "endpoints": len(new_eps),
        "new_edges": len(edges),
        "total_edges": int(total_edges),
    }


def recompute_version_edges(
    conn: sqlite3.Connection, version_id: str, tolerance: float, now: str
) -> int:
    """Full recompute of one version's join edges (e.g. tolerance changed)."""
    drafts = load_active_drafts(conn, version_id)
    placements = _placements_from_drafts(drafts)
    endpoints = load_endpoints(conn, version_id)
    conn.execute("DELETE FROM road_edges WHERE version_id = ?", (version_id,))
    edges = compute_edges_for(
        [ep.id for ep in endpoints], endpoints, placements, tolerance
    )
    _insert_edges(conn, version_id, edges, now)
    return len(edges)


def load_edges(
    conn: sqlite3.Connection, version_id: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT e.id AS edge_id, e.distance, "
        "e.endpoint_a AS a_id, e.endpoint_b AS b_id, "
        "a.tile_id AS a_tile, a.line_id AS a_line, a.end_kind AS a_end, "
        "a.fx AS a_fx, a.fy AS a_fy, "
        "b.tile_id AS b_tile, b.line_id AS b_line, b.end_kind AS b_end, "
        "b.fx AS b_fx, b.fy AS b_fy "
        "FROM road_edges e "
        "JOIN road_endpoints a ON a.id = e.endpoint_a "
        "JOIN road_endpoints b ON b.id = e.endpoint_b "
        "WHERE e.version_id = ? ORDER BY e.id",
        (version_id,),
    ).fetchall()


# --------------------------------------------------------------------------- #
# GeoJSON export (frame coordinates)
# --------------------------------------------------------------------------- #


def _round6(value: float) -> float:
    return round(value, 6)


def build_geojson(
    *,
    version_row: sqlite3.Row,
    drafts: Sequence[sqlite3.Row],
    endpoints: Sequence[Endpoint],
    edges: Sequence[sqlite3.Row],
    tolerance: float,
) -> dict:
    """Road network of one published version, in frame pixel coordinates.

    Fully determined by the stored network state: identical state always
    downloads as identical bytes, so a failed update provably leaves the
    previous network intact.
    """
    joined_ids: set[int] = set()
    for edge in edges:
        joined_ids.add(edge["a_id"])
        joined_ids.add(edge["b_id"])
    features: list[dict] = []

    for draft in drafts:
        lines = parse_draft_obj(json.loads(draft["payload"]))
        for line in lines:
            coords = [
                [_round6(draft["tile_x"] + x), _round6(draft["tile_y"] + y)]
                for x, y in line.points
            ]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "kind": "polyline",
                        "tile_id": draft["tile_id"],
                        "revision": draft["revision"],
                        "line_id": line.line_id,
                        "join_start": line.join_start,
                        "join_end": line.join_end,
                    },
                }
            )

    for edge in edges:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [_round6(edge["a_fx"]), _round6(edge["a_fy"])],
                        [_round6(edge["b_fx"]), _round6(edge["b_fy"])],
                    ],
                },
                "properties": {
                    "kind": "join",
                    "distance": _round6(edge["distance"]),
                    "a": {
                        "tile_id": edge["a_tile"],
                        "line_id": edge["a_line"],
                        "end": edge["a_end"],
                    },
                    "b": {
                        "tile_id": edge["b_tile"],
                        "line_id": edge["b_line"],
                        "end": edge["b_end"],
                    },
                },
            }
        )

    pending = [ep for ep in endpoints if ep.id not in joined_ids]
    for ep in pending:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [_round6(ep.fx), _round6(ep.fy)],
                },
                "properties": {
                    "kind": "pending_endpoint",
                    "tile_id": ep.tile_id,
                    "line_id": ep.line_id,
                    "end": ep.end_kind,
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "roadnet": {
            "version_id": version_row["id"],
            "version_seq": version_row["seq"],
            "canvas": {
                "width": version_row["canvas_width"],
                "height": version_row["canvas_height"],
            },
            "tolerance": tolerance,
            "coordinate_space": (
                "frame pixels (origin top-left, +x right, +y down)"
            ),
            "counts": {
                "polylines": sum(1 for f in features
                                 if f["properties"]["kind"] == "polyline"),
                "joins": sum(1 for f in features
                             if f["properties"]["kind"] == "join"),
                "pending_endpoints": len(pending),
            },
        },
        "features": features,
    }
