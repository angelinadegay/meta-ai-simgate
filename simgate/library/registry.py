"""Filesystem-backed registry for robot and policy library cards."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from .. import presets
from .cards import PolicyCard, RobotCard, WorldCard

CardType = TypeVar("CardType", RobotCard, PolicyCard)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = Path(__file__).resolve().parent / "presets"
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"


class LibraryRegistry:
    def __init__(
        self,
        project_root: Path = PROJECT_ROOT,
        presets_dir: Path = PRESETS_DIR,
        uploads_dir: Path = UPLOADS_DIR,
    ) -> None:
        self.project_root = project_root.resolve()
        self.presets_dir = presets_dir
        self.uploads_dir = uploads_dir

    def robots(self) -> list[RobotCard]:
        return self._cards(RobotCard, "robots")

    def policies(self) -> list[PolicyCard]:
        return self._cards(PolicyCard, "policies")

    def compatible_policies(self, robot_id: str) -> list[PolicyCard]:
        robots = {card.id: card for card in self.robots()}
        robot = robots.get(robot_id)
        if robot is None:
            raise ValueError(f"Unknown robot library id: {robot_id}")
        return [
            policy
            for policy in self.policies()
            if policy.robot_id == robot.id
            or (
                len(policy.joint_order) == len(robot.joint_names)
                and all(name in robot.joint_names for name in policy.joint_order)
            )
        ]

    def worlds(self) -> list[WorldCard]:
        surfaces = presets.SURFACE_FRICTION
        return [
            WorldCard(
                id=f"{body}-{surface}",
                name=f"{body.title()} · {surface.title()}",
                body=body,
                gravity_m_s2=gravity,
                surface=surface,
                friction=friction,
                credit="Reference values in simgate/presets.py; friction values are engineering estimates.",
            )
            for body, gravity in presets.GRAVITY.items()
            for surface, friction in surfaces.items()
        ]

    def _cards(self, card_type: type[CardType], category: str) -> list[CardType]:
        cards: dict[str, CardType] = {}
        for folder in (self.presets_dir / category, self.uploads_dir / category):
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.json")):
                try:
                    card = card_type.model_validate_json(path.read_text(encoding="utf-8"))
                except (OSError, ValidationError, ValueError) as exc:
                    raise ValueError(f"Invalid {category[:-1]} library card at {path}: {exc}") from exc
                if card.id in cards:
                    raise ValueError(f"Duplicate {category[:-1]} library id: {card.id}")
                asset_path = getattr(card, "mjcf_path", None) or getattr(card, "file", None)
                if asset_path is not None:
                    self._validate_asset_exists(asset_path, path)
                cards[card.id] = card
        return list(cards.values())

    def _validate_asset_exists(self, asset_path: str, card_path: Path) -> None:
        candidate = (self.project_root / asset_path).resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"Library asset escapes the project root in {card_path}: {asset_path}") from exc
        if not candidate.is_file():
            raise ValueError(f"Library asset for {card_path} does not exist: {asset_path}")
