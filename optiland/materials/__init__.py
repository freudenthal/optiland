"""This package defines material types used in Optiland, including ideal
materials, materials based on Abbe numbers, and materials loaded from
refractiveindex.info data files."""

from __future__ import annotations

from .abbe import AbbeMaterial, AbbeMaterialE
from .anisotropic import (
    BaseTensorMaterial,
    BianisotropicMaterial,
    BiaxialMaterial,
    QuartzMaterial,
    TensorMaterial,
    UniaxialMaterial,
    alpha_from_gyration,
    euler_zxz_matrix,
    gyration_from_alpha,
    kappa_from_rotatory_power,
    quartz_alpha,
    quartz_rotatory_power,
    standard_air_index,
)
from .base import BaseMaterial
from .catalog import MaterialCatalog
from .ideal import IdealMaterial
from .material import Material
from .material_file import MaterialFile
from .material_spec import MatchPolicy, MaterialSpec
from .material_utils import (
    downsample_glass_map,
    find_closest_glass,
    get_nd_vd,
    get_neighbour_glasses,
    glasses_selection,
    plot_glass_map,
    plot_nk,
)
from .registry import MaterialRegistry
from .warnings import OptilandMaterialWarning

__all__ = [
    # From abbe.py
    "AbbeMaterial",
    "AbbeMaterialE",
    # From anisotropic.py
    "BaseTensorMaterial",
    "BianisotropicMaterial",
    "BiaxialMaterial",
    "QuartzMaterial",
    "TensorMaterial",
    "UniaxialMaterial",
    "alpha_from_gyration",
    "euler_zxz_matrix",
    "gyration_from_alpha",
    "kappa_from_rotatory_power",
    "quartz_alpha",
    "quartz_rotatory_power",
    "standard_air_index",
    # From base.py
    "BaseMaterial",
    # From ideal.py
    "IdealMaterial",
    # From material.py
    "Material",
    # From material_file.py
    "MaterialFile",
    # From catalog.py
    "MaterialCatalog",
    # From material_spec.py
    "MatchPolicy",
    "MaterialSpec",
    # From material_utils.py
    "downsample_glass_map",
    "get_nd_vd",
    "get_neighbour_glasses",
    "glasses_selection",
    "plot_glass_map",
    "plot_nk",
    find_closest_glass,
    # From registry.py
    "MaterialRegistry",
    # From warnings.py
    "OptilandMaterialWarning",
]
