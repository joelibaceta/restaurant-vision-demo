# logic/models.py — Modelos de datos y parámetros
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from shapely.geometry import Polygon


@dataclass
class Mesa:
    id: str
    polygon: List[Tuple[int, int]]
    iop_thr: float = 0.12         # umbral de intersección (IoP) mesa-persona
    y_band: Optional[Tuple[int,int]] = None  # banda vertical (opcional)
    poly: Polygon = field(init=False)
    hist: List[bool] = field(default_factory=list)
    occupied: bool = False
    people_seated: int = 0
    seated_tracks: Dict[int, Dict] = field(default_factory=dict)  # por track_id
    staff_tracks: Set[int] = field(default_factory=set)  # tracks de staff (personas paradas)
    tracks_in_area: Set[int] = field(default_factory=set)  # tracks de clientes en área

    def __post_init__(self):
        self.poly = Polygon([(float(x), float(y)) for x, y in self.polygon])


@dataclass
class LogicParams:
    conf_thr: float = 0.5              # Confianza balanceada
    min_bbox_frac: float = 0.001       # Área mínima original
    max_bbox_frac: float = 0.09        # Área máxima balanceada
    v_thr_px_s: float = 32.0           # Umbral de velocidad balanceado
    sit_seconds: float = 2.0           # Tiempo mínimo balanceado
    hist_frames: int = 6               # Frames para histéresis balanceado
    ttl_lost: float = 11.0             # Tiempo balanceado para mantener track perdido
    min_aspect_ratio: float = 0.35     # Ratio mínimo más permisivo
    max_aspect_ratio: float = 2.9      # Ratio máximo balanceado
    center_weight: float = 0.75        # Peso del centro balanceado
    min_stability_time: float = 1.8    # Tiempo de estabilidad balanceado
    max_displacement_px: float = 75.0  # Desplazamiento balanceado
