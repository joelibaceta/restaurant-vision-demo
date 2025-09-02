# logic.py — Wrapper de compatibilidad para la lógica refactorizada
# Este archivo mantiene compatibilidad con el código existente

from logic.models import Mesa, LogicParams
from logic.occupancy_engine import OccupancyEngine

__all__ = ['Mesa', 'LogicParams', 'OccupancyEngine']
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from shapely.geometry import Polygon, Point

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

class OccupancyEngine:
    def __init__(self, mesas: List[Mesa], frame_size: Tuple[int,int], exclusions: List[List[Tuple[int,int]]] = None, params: LogicParams = None, detector = None):
        self.mesas = mesas
        self.h, self.w = frame_size
        self.frame_area = self.h * self.w
        self.exclusions = [Polygon([(float(x), float(y)) for x,y in poly]) for poly in (exclusions or [])]
        self.params = params or LogicParams()
        self.detector = detector  # Para usar YOLO Pose en clasificación

    def _valid_person(self, det):
        """Validar si una detección es una persona válida para análisis de ocupación"""
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
        
        # Filtro de posición (evitar detecciones en bordes)
        if x1 <= 5 or y1 <= 5 or x2 >= (self.w - 5) or y2 >= (self.h - 5):
            return False
        
        return True

    def _classify_person_in_table_area(self, track, mesa) -> Tuple[str, Dict]:
        """
        Clasifica a una persona como 'customer' o 'excluded'
        Solo procesa clientes sentados, sin clasificación de staff.
        Returns: (classification, metrics_dict)
        """
        x1, y1, x2, y2 = track.xyxy
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        aspect_ratio = h / max(w, 1e-6)
        
        # 1. CREAR RECTÁNGULO DE LA PERSONA
        person_rect = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        
        # Calcular intersección con el polígono de la mesa
        try:
            intersection = person_rect.intersection(mesa.poly)
            intersection_area = intersection.area if intersection.is_valid else 0
        except:
            intersection_area = 0
        
        person_area = person_rect.area
        if person_area == 0:
            return "excluded", {"reason": "invalid_bbox"}
        
        # Porcentaje del área de la persona que está dentro del polígono
        area_percentage_in_polygon = intersection_area / person_area
        
        # 2. VERIFICAR SI TIENE SUFICIENTE ÁREA EN EL POLÍGONO
        min_area_percentage = 0.08  # 8% mínimo - más inclusivo para bordes de mesa
        if area_percentage_in_polygon < min_area_percentage:
            return "excluded", {
                "reason": "insufficient_area", 
                "area_percentage": area_percentage_in_polygon,
                "required": min_area_percentage
            }
        
        # 3. VERIFICAR PARTES DEL CUERPO
        head_y = y1 + h * 0.15      # 15% desde arriba (cabeza)
        torso_y = y1 + h * 0.4      # 40% desde arriba (torso)
        
        head_point = Point(cx, head_y)
        torso_point = Point(cx, torso_y)
        
        head_in_roi = mesa.poly.contains(head_point)
        torso_in_roi = mesa.poly.contains(torso_point)
        
        # 4. FILTRO DE VELOCIDAD: Filtrar peatones muy rápidos
        avg_speed = getattr(track, 'avg_speed', 0)
        if avg_speed > 80.0:  # px/s - peatones muy rápidos
            return "excluded", {
                "reason": "moving_too_fast", 
                "speed": avg_speed,
                "area_percentage": area_percentage_in_polygon
            }
        
        # 5. BANDA VERTICAL (SI APLICA)
        in_y_band = True
        if mesa.y_band:
            y_min, y_max = mesa.y_band
            in_y_band = y_min <= cy <= y_max
        
        if not in_y_band:
            return "excluded", {"reason": "outside_y_band"}
        
                # 6. FILTRAR PERSONAS DE PIE O CON PIES VISIBLES - NO CONTAR COMO CUSTOMERS
        # Usar YOLO Pose para detectar personas paradas con pies visibles
        is_standing_with_feet = False
        if self.detector and hasattr(self.detector, 'is_person_standing_with_feet_visible'):
            try:
                # Usar el detector YOLO Pose real para análisis de postura
                bbox = [x1, y1, x2, y2]
                is_standing_with_feet = self.detector.is_person_standing_with_feet_visible(self.frame, bbox)
            except Exception as e:
                print(f"Error YOLO Pose: {e}, usando fallback geométrico")
                is_standing_with_feet = aspect_ratio > 2.0  # MENOS AGRESIVO (era 1.8)
            else:
                # Fallback geométrico MÁS CONSERVADOR para personas de pie
                is_standing_with_feet = aspect_ratio > 2.5  # MÁS ESTRICTO (era 2.0)        # También filtrar por velocidad: personas caminando no son customers
        is_walking = avg_speed > 12.0  # MENOS AGRESIVO: 12 px/s (era 10)

        if is_standing_with_feet or is_walking:
            classification_type = "staff" if is_standing_with_feet else "excluded"
            return classification_type, {
                "reason": "person_standing_or_walking", 
                "aspect_ratio": aspect_ratio,
                "speed": avg_speed,
                "area_percentage": area_percentage_in_polygon,
                "yolo_pose_detected": is_standing_with_feet if self.detector else False
            }

        # 7. CLIENTE (personas sentadas y quietas)
        # Solo considerar personas con características de sentadas Y sin pies visibles
        is_likely_seated = aspect_ratio <= 2.0 and avg_speed < 12.0  # MENOS ESTRICTO (era 1.8 y 10.0)

        if is_likely_seated and area_percentage_in_polygon > 0.08:
            
            # 8. VALIDACIÓN SECUNDARIA CON YOLO: Verificar que el segmento contenga una persona completa
            if self.detector and self.frame is not None and hasattr(self.detector, 'validate_person_segment'):
                bbox = [x1, y1, x2, y2]
                is_complete_person = self.detector.validate_person_segment(self.frame, bbox)
                
                if not is_complete_person:
                    print(f"🚫 Mesa {mesa.id}: Rechazado track {track.id} - segmento no contiene persona completa (AR: {aspect_ratio:.2f}, Speed: {avg_speed:.1f})")
                    return "excluded", {
                        "reason": "segment_validation_failed",
                        "area_percentage": area_percentage_in_polygon,
                        "aspect_ratio": aspect_ratio,
                        "speed": avg_speed,
                        "yolo_segment_validation": False
                    }
            
            # 9. DOUBLE CHECK: ANÁLISIS ESPECÍFICO DEL POLÍGONO DE MESA
            # Verificar si dentro del polígono solo hay pies/piernas usando YOLO Pose
            if self.detector and self.frame is not None:
                mesa_polygon_check = self._double_check_mesa_polygon(track, mesa)
                if not mesa_polygon_check["valid"]:
                    # Agregar información extra para debug
                    print(f"🚫 Mesa {mesa.id}: Rechazado track {track.id} - {mesa_polygon_check['reason']} (AR: {aspect_ratio:.2f}, Speed: {avg_speed:.1f})")
                    return "excluded", {
                        "reason": "mesa_polygon_check_failed",
                        "mesa_check_reason": mesa_polygon_check["reason"],
                        "area_percentage": area_percentage_in_polygon,
                        "aspect_ratio": aspect_ratio,
                        "speed": avg_speed
                    }
                else:
                    # Log casos aceptados para debug
                    print(f"✅ Mesa {mesa.id}: Aceptado track {track.id} - {mesa_polygon_check['reason']} (AR: {aspect_ratio:.2f}, Speed: {avg_speed:.1f})")
            
            return "customer", {
                "area_percentage": area_percentage_in_polygon,
                "head_in_roi": head_in_roi,
                "torso_in_roi": torso_in_roi,
                "aspect_ratio": aspect_ratio,
                "speed": avg_speed,
                "yolo_segment_validation": True
            }
        
        # 8. CASOS AMBIGUOS -> EXCLUIR
        return "excluded", {
            "reason": "ambiguous_classification",
            "area_percentage": area_percentage_in_polygon,
            "aspect_ratio": aspect_ratio,
            "speed": avg_speed
        }

    def _double_check_mesa_polygon(self, track, mesa) -> Dict:
        """
        DOUBLE CHECK: Analiza SOLO la región del polígono de la mesa para verificar
        que no estemos contando solo pies/piernas como clientes.
        
        Returns: {"valid": bool, "reason": str}
        """
        try:
            x1, y1, x2, y2 = track.xyxy
            
            # Obtener bounding box del polígono de la mesa
            mesa_coords = mesa.polygon
            mesa_x_coords = [coord[0] for coord in mesa_coords]
            mesa_y_coords = [coord[1] for coord in mesa_coords]
            
            mesa_x_min = max(0, int(min(mesa_x_coords)))
            mesa_y_min = max(0, int(min(mesa_y_coords)))
            mesa_x_max = min(self.frame.shape[1], int(max(mesa_x_coords)))
            mesa_y_max = min(self.frame.shape[0], int(max(mesa_y_coords)))
            
            # Validar que el recorte sea válido
            if mesa_x_max <= mesa_x_min or mesa_y_max <= mesa_y_min:
                return {"valid": True, "reason": "invalid_mesa_bounds"}
            
            # Extraer solo la región del polígono de la mesa
            mesa_roi = self.frame[mesa_y_min:mesa_y_max, mesa_x_min:mesa_x_max]
            
            if mesa_roi.size == 0:
                return {"valid": True, "reason": "empty_mesa_roi"}
            
            # Ajustar coordenadas de la persona al ROI de la mesa
            person_x1_roi = max(0, x1 - mesa_x_min)
            person_y1_roi = max(0, y1 - mesa_y_min)
            person_x2_roi = min(mesa_roi.shape[1], x2 - mesa_x_min)
            person_y2_roi = min(mesa_roi.shape[0], y2 - mesa_y_min)
            
            # Verificar que la persona esté dentro del ROI
            if person_x2_roi <= person_x1_roi or person_y2_roi <= person_y1_roi:
                return {"valid": True, "reason": "person_outside_mesa_roi"}
            
            # Usar el detector para analizar específicamente esta región
            if hasattr(self.detector, '_has_head_or_torso_in_mesa_roi'):
                has_valid_body_parts = self.detector._has_head_or_torso_in_mesa_roi(
                    mesa_roi, 
                    person_x1_roi, person_y1_roi, 
                    person_x2_roi, person_y2_roi
                )
                
                if not has_valid_body_parts:
                    return {"valid": False, "reason": "only_feet_in_mesa_polygon"}
                
                return {"valid": True, "reason": "valid_body_parts_detected"}
            else:
                # Fallback: aceptar si no hay detector
                return {"valid": True, "reason": "no_pose_detector"}
                
        except Exception as e:
            # En caso de error, ser conservador y aceptar
            print(f"Error en double check de mesa: {e}")
            return {"valid": True, "reason": f"error_in_check: {str(e)}"}

    def step(self, tracks, frame=None) -> None:
        """Actualiza mesas usando tracks (Track) con .id .xyxy .cx .cy .speed .last_t"""
        now = time.time()
        
        # Guardar frame para YOLO Pose analysis
        self.frame = frame
        
        # Primero, filtrar tracks válidos
        valid_tracks = []
        rejected_tracks = []
        for tr in tracks:
            det_dict = {"xyxy": tr.xyxy}
            if self._valid_person(det_dict):
                valid_tracks.append(tr)
            else:
                rejected_tracks.append(tr)
        
        # Debug: mostrar tracks rechazados (solo ocasionalmente)
        if len(rejected_tracks) > 0 and len(tracks) > 0:
            rejection_rate = len(rejected_tracks) / len(tracks)
            if rejection_rate > 0.5:  # Solo mostrar si rechazo > 50%
                print(f"⚠️  Muchos tracks rechazados: {len(rejected_tracks)}/{len(tracks)} ({rejection_rate:.1%})")
                for tr in rejected_tracks[:2]:  # Solo mostrar primeros 2
                    x1, y1, x2, y2 = tr.xyxy
                    w, h = x2 - x1, y2 - y1
                    area_frac = (w * h) / self.frame_area
                    aspect_ratio = h / max(w, 1e-6)
                    print(f"  Track {tr.id}: área={area_frac:.4f}, aspect={aspect_ratio:.2f}")
        
        # Debug: mostrar todos los tracks válidos
        valid_track_ids = [tr.id for tr in valid_tracks]
        
        # Actualizar información de cada mesa con tracks específicos para esa mesa
        for mesa in self.mesas:
            seated_now: Set[int] = set()
            candidates_in_area: Dict[int, Dict] = {}
            tracks_in_area = 0

            # Filtrar tracks para esta mesa y clasificar
            tracks_for_this_mesa = []
            staff_tracks_for_this_mesa = set()
            
            for tr in valid_tracks:
                # Verificar si este track intersecta con esta mesa específica
                classification, metrics = self._classify_person_in_table_area(tr, mesa)
                if classification == "customer":  # Solo procesar clientes para seating
                    tracks_for_this_mesa.append((tr, classification, metrics))
                elif classification == "staff":  # Gestionar staff por separado
                    staff_tracks_for_this_mesa.add(tr.id)

            # Actualizar staff_tracks para esta mesa
            mesa.staff_tracks = staff_tracks_for_this_mesa

            # Analizar cada track válido para esta mesa específica
            for tr, classification, metrics in tracks_for_this_mesa:
                tracks_in_area += 1
                # Procesar como cliente (lógica original)
                candidates_in_area[tr.id] = metrics
                
                # Debug para mesa 01
                if mesa.id == "01":
                    # Calcular IoP correctamente para debug
                    person_rect = Polygon([(tr.xyxy[0], tr.xyxy[1]), (tr.xyxy[2], tr.xyxy[1]), (tr.xyxy[2], tr.xyxy[3]), (tr.xyxy[0], tr.xyxy[3])])
                    try:
                        intersection = person_rect.intersection(mesa.poly)
                        intersection_area = intersection.area if intersection.is_valid else 0
                    except:
                        intersection_area = 0
                    iop = intersection_area / max(mesa.poly.area, 1e-6)
                
                # === LÓGICA DE CLIENTE ===
                
                # Obtener o crear estado de esta persona en esta mesa
                st = mesa.seated_tracks.get(tr.id, {
                    "cand_t": now, 
                    "last_t": now, 
                    "seated": False,
                    "stability_start": now,
                    "stable": False,
                    "seated_position": None  # Guardar posición cuando se sienta
                })
                
                # Usar velocidad promedio (más estable) en lugar de instantánea
                current_speed = tr.avg_speed if hasattr(tr, 'avg_speed') else tr.speed
                is_moving_slowly = current_speed < self.params.v_thr_px_s
                
                current_speed = tr.avg_speed if hasattr(tr, 'avg_speed') else tr.speed
                is_moving_slowly = current_speed < self.params.v_thr_px_s
                
                # Si ya está sentado, verificar desplazamiento desde posición original
                if st["seated"] and st["seated_position"]:
                    seated_x, seated_y = st["seated_position"]
                    current_displacement = math.hypot(tr.cx - seated_x, tr.cy - seated_y)
                    
                    # Solo quitar estado sentado si hay desplazamiento significativo
                    if current_displacement > self.params.max_displacement_px:
                        st["seated"] = False
                        st["seated_position"] = None
                        st["stable"] = False
                        st["cand_t"] = now
                    # Si sigue dentro de la zona, mantener como sentado independientemente de velocidad
                
                elif is_moving_slowly:
                    # Si no era estable antes, marcar inicio de estabilidad
                    if not st.get("stable", False):
                        st["stability_start"] = now
                        st["stable"] = True
                    
                    # Verificar si ha estado estable el tiempo suficiente
                    stability_duration = now - st["stability_start"]
                    
                    if stability_duration >= self.params.min_stability_time:
                        # Ha estado estable, puede ser candidato a sentado
                        if not st["seated"] and (now - st["cand_t"]) >= self.params.sit_seconds:
                            st["seated"] = True
                            # Guardar posición cuando se sienta por primera vez
                            st["seated_position"] = (tr.cx, tr.cy)
                else:
                    # Se está moviendo rápido y NO está sentado previamente
                    if not st["seated"]:
                        st["stable"] = False
                        st["cand_t"] = now  # Resetear tiempo de candidatura
                    # Si ya estaba sentado, la lógica de desplazamiento de arriba ya lo maneja

                # Actualizar timestamp
                st["last_t"] = now
                mesa.seated_tracks[tr.id] = st
                
                # Agregar a sentados si está marcado como sentado
                if st["seated"]:
                    seated_now.add(tr.id)

            # Limpiar tracks perdidos de seated_tracks
            for tid, st in list(mesa.seated_tracks.items()):
                time_since_last_seen = now - st["last_t"]
                
                if time_since_last_seen > self.params.ttl_lost:
                    # Track perdido por mucho tiempo
                    was_seated = st.get("seated", False)
                    mesa.seated_tracks.pop(tid, None)
                    if was_seated:
                        print(f"❌ Mesa {mesa.id}: Perdido track {tid} después de {time_since_last_seen:.1f}s")
                elif st.get("seated") and time_since_last_seen <= self.params.ttl_lost:
                    # Mantener personas sentadas aunque no las veamos temporalmente (oclusiones)
                    seated_now.add(tid)

            # Actualizar tracks_in_area para visualización (solo clientes)
            mesa.tracks_in_area = seated_now.copy()

            # Aplicar histéresis para determinar estado de ocupación
            mesa.hist.append(len(seated_now) > 0)
            if len(mesa.hist) > self.params.hist_frames:
                mesa.hist = mesa.hist[-self.params.hist_frames:]
            
            # Una mesa está ocupada si la mayoría de frames recientes tienen personas sentadas
            votes_occupied = sum(mesa.hist)
            was_occupied = mesa.occupied
            mesa.occupied = (votes_occupied >= self.params.hist_frames // 2 + 1)
            mesa.people_seated = len(seated_now)
            
            # Log cambios de estado
            if mesa.occupied != was_occupied:
                pass  # Mesa cambió de estado

    def combine_detections(self, global_dets, roi_dets):
        """
        Combina detecciones globales y de ROI de manera inteligente:
        1. CORRIGE detecciones globales con versiones ROI más precisas
        2. AGREGA personas perdidas que solo se ven en ROI
        3. EVITA duplicados y detecciones de solo pies
        """
        refined_dets = []
        used_roi_indices = set()
        
        # Parámetros
        improvement_threshold = 0.3  # IoU mínimo para considerar mejora
        confidence_boost = 0.1      # Bonus de confianza para detecciones ROI
        
        # 1. PROCESAR DETECCIONES GLOBALES: corregir o mantener
        for global_det in global_dets:
            global_x1, global_y1, global_x2, global_y2 = global_det["xyxy"]
            global_area = (global_x2 - global_x1) * (global_y2 - global_y1)
            global_conf = global_det["conf"]
            
            best_replacement = None
            best_iou = 0
            best_roi_idx = -1
            
            # Buscar detección ROI que mejore esta global
            for idx, roi_det in enumerate(roi_dets):
                if idx in used_roi_indices:
                    continue
                    
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_det["xyxy"]
                roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)
                roi_conf = roi_det["conf"]
                
                # Calcular IoU
                inter_x1 = max(global_x1, roi_x1)
                inter_y1 = max(global_y1, roi_y1)
                inter_x2 = min(global_x2, roi_x2)
                inter_y2 = min(global_y2, roi_y2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    union_area = global_area + roi_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    # ¿Esta detección ROI mejora la global?
                    if iou > improvement_threshold and iou > best_iou:
                        # Criterios de mejora:
                        # 1. Mayor confianza
                        # 2. Mejor proporción (más cuadrada = mejor persona)
                        roi_aspect = (roi_y2 - roi_y1) / max(roi_x2 - roi_x1, 1e-6)
                        global_aspect = (global_y2 - global_y1) / max(global_x2 - global_x1, 1e-6)
                        
                        is_improvement = (
                            roi_conf > global_conf * 0.9 or  # Confianza similar o mejor
                            (1.5 <= roi_aspect <= 3.0 and global_aspect < 1.2)  # Mejor proporción
                        )
                        
                        if is_improvement:
                            best_replacement = roi_det.copy()
                            best_replacement["conf"] = min(1.0, roi_conf + confidence_boost)
                            best_iou = iou
                            best_roi_idx = idx
            
            # Usar la mejor detección (ROI mejorada o global original)
            if best_replacement:
                refined_dets.append(best_replacement)
                used_roi_indices.add(best_roi_idx)
            else:
                refined_dets.append(global_det)
        
        # 2. AGREGAR DETECCIONES ROI NUEVAS (que no mejoraron ninguna global)
        for idx, roi_det in enumerate(roi_dets):
            if idx not in used_roi_indices:
                # Verificar que no se solape significativamente con las ya refinadas
                is_new_person = True
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_det["xyxy"]
                roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)
                
                for refined_det in refined_dets:
                    ref_x1, ref_y1, ref_x2, ref_y2 = refined_det["xyxy"]
                    ref_area = (ref_x2 - ref_x1) * (ref_y2 - ref_y1)
                    
                    # Calcular IoU con detecciones ya refinadas
                    inter_x1 = max(roi_x1, ref_x1)
                    inter_y1 = max(roi_y1, ref_y1)
                    inter_x2 = min(roi_x2, ref_x2)
                    inter_y2 = min(roi_y2, ref_y2)
                    
                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        union_area = roi_area + ref_area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        if iou > 0.3:  # Demasiado solapamiento
                            is_new_person = False
                            break
                
                # Solo agregar si es verdaderamente nueva
                if is_new_person:
                    refined_dets.append(roi_det)
        
        return refined_dets