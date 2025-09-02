# logic/mesa_analyzer.py — Análisis específico de estado de mesas
import time
from typing import Set, Dict, List
from .models import Mesa


class MesaAnalyzer:
    """Maneja el análisis y actualización de estado de cada mesa"""
    
    def __init__(self, params):
        self.params = params
    
    def update_mesa_state(self, mesa: Mesa, tracks_for_mesa: List, now: float):
        """Actualiza el estado completo de una mesa basado en los tracks"""
        seated_now: Set[int] = set()
        candidates_in_area: Dict[int, Dict] = {}
        staff_tracks_for_mesa = set()
        
        # Procesar cada track
        for tr, classification, metrics in tracks_for_mesa:
            if classification == "customer":
                candidates_in_area[tr.id] = metrics
                self._process_customer_track(mesa, tr, now, seated_now)
            elif classification == "staff":
                staff_tracks_for_mesa.add(tr.id)
        
        # Actualizar staff tracks
        mesa.staff_tracks = staff_tracks_for_mesa
        
        # Limpiar tracks perdidos
        self._cleanup_lost_tracks(mesa, now, seated_now)
        
        # Actualizar estado de ocupación con histéresis
        self._update_occupancy_state(mesa, seated_now)
        
        # Actualizar tracks en área para visualización
        mesa.tracks_in_area = seated_now.copy()
        
        return len(seated_now)
    
    def _process_customer_track(self, mesa: Mesa, track, now: float, seated_now: Set[int]):
        """Procesa un track de customer individual"""
        # Obtener o crear estado del track
        track_state = mesa.seated_tracks.get(track.id, {
            "cand_t": now, 
            "last_t": now, 
            "seated": False,
            "stability_start": now,
            "stable": False,
            "seated_position": None
        })
        
        # Analizar movimiento
        current_speed = track.avg_speed if hasattr(track, 'avg_speed') else track.speed
        is_moving_slowly = current_speed < self.params.v_thr_px_s
        
        # Verificar estabilidad si ya está sentado
        if track_state["seated"] and track_state["seated_position"]:
            displacement = self._calculate_displacement(track, track_state["seated_position"])
            if displacement > self.params.max_displacement_px:
                # Se movió demasiado, resetear
                track_state.update({
                    "seated": False,
                    "cand_t": now,
                    "stability_start": now,
                    "stable": False,
                    "seated_position": None
                })
        elif is_moving_slowly:
            # Verificar tiempo de estabilidad
            stability_time = now - track_state["stability_start"]
            
            if not track_state["stable"] and stability_time >= self.params.min_stability_time:
                track_state["stable"] = True
            
            if track_state["stable"]:
                candidate_time = now - track_state["cand_t"]
                if candidate_time >= self.params.sit_seconds:
                    # Marcar como sentado
                    track_state.update({
                        "seated": True,
                        "seated_position": (track.cx, track.cy)
                    })
        else:
            # Se está moviendo, resetear estabilidad
            track_state.update({
                "stability_start": now,
                "stable": False,
                "seated": False,
                "seated_position": None
            })
        
        # Actualizar timestamp
        track_state["last_t"] = now
        mesa.seated_tracks[track.id] = track_state
        
        # Agregar a sentados si está marcado como sentado
        if track_state["seated"]:
            seated_now.add(track.id)
    
    def _calculate_displacement(self, track, seated_position):
        """Calcula desplazamiento desde posición de sentado"""
        if seated_position is None:
            return 0
        
        seated_x, seated_y = seated_position
        current_x, current_y = track.cx, track.cy
        
        dx = current_x - seated_x
        dy = current_y - seated_y
        return (dx * dx + dy * dy) ** 0.5
    
    def _cleanup_lost_tracks(self, mesa: Mesa, now: float, seated_now: Set[int]):
        """Limpia tracks que se han perdido"""
        for tid, state in list(mesa.seated_tracks.items()):
            time_since_last_seen = now - state["last_t"]
            
            if time_since_last_seen > self.params.ttl_lost:
                # Track perdido por mucho tiempo, eliminar
                del mesa.seated_tracks[tid]
            elif state.get("seated") and time_since_last_seen <= self.params.ttl_lost:
                # Track sentado pero temporalmente perdido, mantener
                seated_now.add(tid)
    
    def _update_occupancy_state(self, mesa: Mesa, seated_now: Set[int]):
        """Actualiza estado de ocupación usando histéresis"""
        # Aplicar histéresis
        mesa.hist.append(len(seated_now) > 0)
        if len(mesa.hist) > self.params.hist_frames:
            mesa.hist = mesa.hist[-self.params.hist_frames:]
        
        # Determinar ocupación por mayoría de votos
        votes_occupied = sum(mesa.hist)
        was_occupied = mesa.occupied
        mesa.occupied = (votes_occupied >= self.params.hist_frames // 2 + 1)
        mesa.people_seated = len(seated_now)
        
        # Log cambios de estado si es necesario
        if mesa.occupied != was_occupied:
            status = "ocupada" if mesa.occupied else "libre"
            print(f"📍 Mesa {mesa.id}: {status} ({mesa.people_seated} personas)")
    
    def combine_detections(self, global_dets, roi_dets):
        """
        Combina detecciones globales y de ROI de manera inteligente
        """
        refined_dets = []
        used_roi_indices = set()
        
        # Parámetros
        improvement_threshold = 0.3
        confidence_boost = 0.1
        
        # 1. Procesar detecciones globales
        for global_det in global_dets:
            global_x1, global_y1, global_x2, global_y2 = global_det["xyxy"]
            global_area = (global_x2 - global_x1) * (global_y2 - global_y1)
            global_conf = global_det["conf"]
            
            best_replacement = None
            best_iou = 0
            best_roi_idx = -1
            
            # Buscar ROI que mejore esta detección global
            for idx, roi_det in enumerate(roi_dets):
                if idx in used_roi_indices:
                    continue
                    
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_det["xyxy"]
                roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)
                roi_conf = roi_det["conf"]
                
                # Calcular IoU
                iou = self._calculate_iou(global_det["xyxy"], roi_det["xyxy"])
                
                if iou > improvement_threshold:
                    # Determinar si ROI es mejor
                    confidence_improvement = roi_conf > global_conf
                    area_improvement = abs(roi_area - global_area) < global_area * 0.2
                    
                    if confidence_improvement and area_improvement and iou > best_iou:
                        best_replacement = {
                            "xyxy": roi_det["xyxy"],
                            "conf": roi_conf + confidence_boost
                        }
                        best_iou = iou
                        best_roi_idx = idx
            
            # Usar la mejor detección
            if best_replacement:
                refined_dets.append(best_replacement)
                used_roi_indices.add(best_roi_idx)
            else:
                refined_dets.append(global_det)
        
        # 2. Agregar detecciones ROI nuevas
        for idx, roi_det in enumerate(roi_dets):
            if idx not in used_roi_indices:
                enhanced_det = {
                    "xyxy": roi_det["xyxy"],
                    "conf": roi_det["conf"] + confidence_boost
                }
                refined_dets.append(enhanced_det)
        
        return refined_dets
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calcula Intersection over Union entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersección
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Unión
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
