"""Saved search presets (per-user named filter/context selections)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import SavedSearch, User
from ..schemas import PresetCreate, PresetOut

router = APIRouter(prefix="/presets", tags=["presets"])


def _to_out(p: SavedSearch) -> PresetOut:
    return PresetOut(id=p.id, name=p.name, params=p.params, created_at=p.created_at)


@router.post("", response_model=PresetOut)
def create_preset(
    body: PresetCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PresetOut:
    preset = SavedSearch(user_id=user.id, name=body.name, params=body.params)
    db.add(preset)
    db.commit()
    db.refresh(preset)
    return _to_out(preset)


@router.get("", response_model=list[PresetOut])
def list_presets(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[PresetOut]:
    rows = db.scalars(
        select(SavedSearch)
        .where(SavedSearch.user_id == user.id)
        .order_by(SavedSearch.created_at.desc())
    ).all()
    return [_to_out(p) for p in rows]


@router.delete("/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_preset(
    preset_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    preset = db.get(SavedSearch, preset_id)
    if preset is None or preset.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "preset not found")
    db.delete(preset)
    db.commit()
