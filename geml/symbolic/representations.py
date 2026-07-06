"""Shared symbolic representation mode names."""

from __future__ import annotations

from typing import Literal

type RepresentationMode = Literal[
    "ast",
    "restricted_eml_pure",
    "restricted_eml_with_derived",
]
type EmlRepresentationMode = Literal["restricted_eml_pure", "restricted_eml_with_derived"]

EML_REPRESENTATION_MODES: tuple[EmlRepresentationMode, ...] = (
    "restricted_eml_pure",
    "restricted_eml_with_derived",
)
REPRESENTATION_MODES: tuple[RepresentationMode, ...] = (
    "ast",
    *EML_REPRESENTATION_MODES,
)
