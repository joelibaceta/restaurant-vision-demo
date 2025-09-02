# logic/occupancy_engine.py — Motor principal de análisis de ocupación
import time
from typing import List, Tuple, Set
from shapely.geometry import Polygon
from .models import Mesa, LogicParams
from .person_classifier import PersonClassifier
from .mesa_analyzer import MesaAnalyzer


class OccupancyEngine:
    """Motor principal que coordina el análisis de ocupación"""
    
    def __init__(self, mesas: List[Mesa], frame_size: Tuple[int,int], 
                 exclusions: List[List[Tuple[int,int]]] = None, 
                 params: LogicParams = None, detector = None):
        self.mesas = mesas
        self.h, self.w = frame_size
        self.frame_area = self.h * self.w
        self.exclusions = [Polygon([(float(x), float(y)) for x,y in poly]) 
                          for poly in (exclusions or [])]
        self.params = params or LogicParams()
        
        # Inicializar componentes especializados
        self.person_classifier = PersonClassifier(detector, self.w, self.h)
        self.mesa_analyzer = MesaAnalyzer(self.params)
        
        # Frame actual para análisis
        self.current_frame = None

    def step(self, tracks, frame=None) -> None:
        """Actualiza mesas usando tracks con análisis completo"""
        now = time.time()
        
        # Actualizar frame actual
        self.current_frame = frame
        self.person_classifier.set_current_frame(frame)
        
        # Filtrar tracks válidos
        valid_tracks = self._filter_valid_tracks(tracks)
        
        # Procesar cada mesa
        for mesa in self.mesas:
            # Clasificar tracks para esta mesa
            tracks_for_mesa = self._classify_tracks_for_mesa(valid_tracks, mesa)
            
            # Actualizar estado de la mesa
            self.mesa_analyzer.update_mesa_state(mesa, tracks_for_mesa, now)

    def _filter_valid_tracks(self, tracks):
        """Filtra tracks válidos para análisis"""
        valid_tracks = []
        rejected_tracks = []
        
        for tr in tracks:
            det_dict = {"xyxy": tr.xyxy}
            if self._is_valid_person(det_dict):
                valid_tracks.append(tr)
            else:
                rejected_tracks.append(tr)
        
        # Debug ocasional de tracks rechazados
        self._debug_rejected_tracks(tracks, rejected_tracks)
        
        return valid_tracks

    def _is_valid_person(self, det):
        """Validar si una detección es una persona válida"""
        x1, y1, x2, y2 = det["xyxy"]
        w, h = x2 - x1, y2 - y1
        area = w * h
        
        # Filtros básicos de tamaño
        min_area = self.params.min_bbox_frac * self.frame_area
        max_area = self.params.max_bbox_frac * self.frame_area
        if not (min_area <= area <= max_area):
            return False
        
        # Filtro de aspect ratio
        aspect_ratio = h / max(w, 1e-6)
        if not (self.params.min_aspect_ratio <= aspect_ratio <= self.params.max_aspect_ratio):
            return False
        
        # Filtro de posición (evitar bordes)
        if x1 <= 5 or y1 <= 5 or x2 >= (self.w - 5) or y2 >= (self.h - 5):
            return False
        
        return True

    def _classify_tracks_for_mesa(self, valid_tracks, mesa):
        """Clasifica todos los tracks válidos para una mesa específica"""
        tracks_for_mesa = []
        
        for tr in valid_tracks:
            # Clasificar este track para esta mesa
            classification, metrics = self.person_classifier.classify_person_in_table_area(tr, mesa)
            
            if classification in ["customer", "staff"]:
                tracks_for_mesa.append((tr, classification, metrics))
                
                # Debug para mesa específica
                if mesa.id == "01" and classification == "customer":
                    print(f"✅ Mesa {mesa.id}: Track {tr.id} clasificado como {classification}")
        
        return tracks_for_mesa

    def _debug_rejected_tracks(self, all_tracks, rejected_tracks):
        """Debug de tracks rechazados"""
        if len(rejected_tracks) > 0 and len(all_tracks) > 0:
            rejection_rate = len(rejected_tracks) / len(all_tracks)
            if rejection_rate > 0.5:  # Solo mostrar si rechazo > 50%
                print(f"⚠️  Muchos tracks rechazados: {len(rejected_tracks)}/{len(all_tracks)} ({rejection_rate:.1%})")
                for tr in rejected_tracks[:2]:  # Solo mostrar primeros 2
                    x1, y1, x2, y2 = tr.xyxy
                    w, h = x2 - x1, y2 - y1
                    area_frac = (w * h) / self.frame_area
                    aspect_ratio = h / max(w, 1e-6)
                    print(f"  Track {tr.id}: área={area_frac:.4f}, aspect={aspect_ratio:.2f}")

    def combine_detections(self, global_dets, roi_dets):
        """Combina detecciones globales y ROI (delega al mesa analyzer)"""
        return self.mesa_analyzer.combine_detections(global_dets, roi_dets)
