"""Restricted EML tree data structures."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from geml.symbolic.metrics import TreeStatistics
from geml.symbolic.representations import EmlRepresentationMode

type MetadataValue = str | int | float | bool | list[str]

EmlNodeKind = Literal["eml", "variable", "constant", "derived"]


class EmlNode(BaseModel):
    """A node in a restricted EML binary tree."""

    id: int = Field(ge=0)
    label: str
    kind: EmlNodeKind
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class EmlEdge(BaseModel):
    """A directed parent-to-child EML edge."""

    source: int = Field(ge=0)
    target: int = Field(ge=0)
    position: int = Field(ge=0, le=1)


class EmlTree(BaseModel):
    """Serializable restricted EML tree representation."""

    representation_mode: EmlRepresentationMode
    nodes: list[EmlNode]
    edges: list[EmlEdge]
    root_id: int
    node_labels: dict[int, str]
    metadata: dict[str, MetadataValue]
    statistics: TreeStatistics
    normal_leaf_count: int = Field(ge=0)
    derived_leaf_count: int = Field(ge=0)
    hidden_compound_leaf_count: int = Field(ge=0)
    ast_statistics: TreeStatistics
    alpha: float | None
    alpha_valid: bool
