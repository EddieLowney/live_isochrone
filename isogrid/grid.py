"""Sampling geometry.

A Grid turns an origin plus a few scale parameters into an ordered set of cells.
Nothing downstream knows the tiling is hexagonal, so swapping in H3 later means
writing an H3Grid that yields the same Cell objects.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .registry import GRIDS
from .types import Cell, Coord

SQRT3 = math.sqrt(3.0)


class Grid(ABC):
    """Ordered collection of sampling cells around an origin."""

    def __init__(self, origin: Coord) -> None:
        self.origin = origin
        self._cells: list[Cell] = []
        self._index: dict[str, Cell] = {}

    def _finalise(self, cells: list[Cell]) -> None:
        self._cells = cells
        self._index = {c.id: c for c in cells}

    @property
    def cells(self) -> list[Cell]:
        return self._cells

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self) -> Iterator[Cell]:
        return iter(self._cells)

    def get(self, cell_id: str) -> Cell:
        return self._index[cell_id]

    def has(self, cell_id: str) -> bool:
        return cell_id in self._index

    @abstractmethod
    def neighbours(self, cell_id: str) -> list[str]:
        """Adjacent cell ids. Used by spatial reconstructors and by any future
        edge-aware smoothing."""

    def describe(self) -> dict:
        return {
            "type": getattr(self, "registry_name", type(self).__name__),
            "cells": len(self),
            "origin": {"lat": self.origin.lat, "lon": self.origin.lon},
        }


@GRIDS.register("hex")
class HexGrid(Grid):
    """Pointy-top hexagons in axial coordinates, laid out in concentric rings.

    `cell_km` is the circumradius (centre to vertex). `rings` counts rings
    outward from the origin cell, so total cells is 3n(n+1)+1 — 7 for one ring,
    61 for four, 217 for eight. That formula is the knob for scale.
    """

    AXIAL_DIRS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))

    def __init__(self, origin: Coord, cell_km: float = 2.0, rings: int = 3) -> None:
        super().__init__(origin)
        self.cell_km = float(cell_km)
        self.rings = int(rings)
        cells: list[Cell] = []
        for q in range(-self.rings, self.rings + 1):
            lo = max(-self.rings, -q - self.rings)
            hi = min(self.rings, -q + self.rings)
            for r in range(lo, hi + 1):
                cells.append(self._make_cell(q, r))
        cells.sort(key=lambda c: (c.ring, c.id))
        self._finalise(cells)

    def _axial_to_km(self, q: int, r: int) -> tuple[float, float]:
        east = self.cell_km * SQRT3 * (q + r / 2.0)
        north = -self.cell_km * 1.5 * r
        return north, east

    def _make_cell(self, q: int, r: int) -> Cell:
        north, east = self._axial_to_km(q, r)
        centre = self.origin.offset_km(north, east)
        # Vertices at compass bearings 0, 60, ... 300 put a vertex due north,
        # which is what "pointy-top" means and what the axial spacing above
        # assumes. Offsetting these by 30 degrees draws a flat-top hexagon on a
        # pointy-top lattice: the cells then overlap along one axis and leave
        # gaps along the other.
        poly = tuple(
            centre.offset_km(
                self.cell_km * math.cos(math.radians(60 * i)),
                self.cell_km * math.sin(math.radians(60 * i)),
            )
            for i in range(6)
        )
        ring = (abs(q) + abs(q + r) + abs(r)) // 2
        return Cell(id=f"h:{q}:{r}", center=centre, polygon=poly, ring=ring)

    @staticmethod
    def _parse(cell_id: str) -> tuple[int, int]:
        _, q, r = cell_id.split(":")
        return int(q), int(r)

    def neighbours(self, cell_id: str) -> list[str]:
        q, r = self._parse(cell_id)
        out = []
        for dq, dr in self.AXIAL_DIRS:
            nid = f"h:{q + dq}:{r + dr}"
            if nid in self._index:
                out.append(nid)
        return out

    def describe(self) -> dict:
        d = super().describe()
        d.update({"cell_km": self.cell_km, "rings": self.rings})
        return d


@GRIDS.register("square")
class SquareGrid(Grid):
    """Regular lattice. Present mainly to prove the abstraction holds — hex is
    the better default because every neighbour is equidistant."""

    def __init__(self, origin: Coord, cell_km: float = 2.0, rings: int = 3) -> None:
        super().__init__(origin)
        self.cell_km = float(cell_km)
        self.rings = int(rings)
        cells = []
        for i in range(-self.rings, self.rings + 1):
            for j in range(-self.rings, self.rings + 1):
                centre = self.origin.offset_km(-j * self.cell_km, i * self.cell_km)
                h = self.cell_km / 2.0
                poly = (
                    centre.offset_km(h, -h),
                    centre.offset_km(h, h),
                    centre.offset_km(-h, h),
                    centre.offset_km(-h, -h),
                )
                cells.append(
                    Cell(f"s:{i}:{j}", centre, poly, ring=max(abs(i), abs(j)))
                )
        cells.sort(key=lambda c: (c.ring, c.id))
        self._finalise(cells)

    def neighbours(self, cell_id: str) -> list[str]:
        _, i, j = cell_id.split(":")
        i, j = int(i), int(j)
        out = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nid = f"s:{i + di}:{j + dj}"
            if nid in self._index:
                out.append(nid)
        return out


def build_grid(kind: str, origin: Coord, **kwargs) -> Grid:
    return GRIDS.build(kind, origin=origin, **kwargs)
