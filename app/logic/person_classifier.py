# logic/person_classifier.py — Clasificación de personas (customer/staff/excluded)
from typing import Tuple, Dict
from shapely.geometry import Polygon, Point


class PersonClassifier:
    """Clasifica personas en customer, staff o excluded"""
    
    def __init__(self, detector=None, frame_w=1280, frame_h=720):
        self.detector = detector
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.current_frame = None
    
    def set_current_frame(self, frame):
        """Actualiza el frame actual para análisis"""
        self.current_frame = frame
    
    def classify_person_in_table_area(self, track, mesa) -> Tuple[str, Dict]:
        """
        Clasifica a una persona como 'customer', 'staff' o 'excluded'
        Returns: (classification, metrics_dict)
        """
        x1, y1, x2, y2 = track.xyxy
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        aspect_ratio = h / max(w, 1e-6)
        
        # 1. Crear rectángulo de la persona
        person_rect = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        
        # 2. Calcular intersección con mesa
        intersection_area = self._calculate_intersection_area(person_rect, mesa.poly)
        person_area = person_rect.area
        
        if person_area == 0:
            return "excluded", {"reason": "invalid_bbox"}
        
        area_percentage_in_polygon = intersection_area / person_area
        
        # 3. Filtros básicos
        basic_filters_result = self._apply_basic_filters(
            area_percentage_in_polygon, track, mesa, cx, cy
        )
        if basic_filters_result is not None:
            return basic_filters_result
        
        # 4. Análisis de postura (standing vs seated)
        posture_result = self._analyze_posture(track, aspect_ratio, area_percentage_in_polygon)
        if posture_result is not None:
            return posture_result
        
        # 5. Validación de segmento para personas sentadas
        if self._is_likely_seated(aspect_ratio, track):
            segment_validation_result = self._validate_customer_segment(track, area_percentage_in_polygon)
            if segment_validation_result is not None:
                return segment_validation_result
            
            # 6. Double check del polígono de mesa
            mesa_check_result = self._double_check_mesa_polygon(track, mesa)
            if not mesa_check_result["valid"]:
                return "excluded", {
                    "reason": "mesa_polygon_check_failed",
                    "mesa_check_reason": mesa_check_result["reason"],
                    "area_percentage": area_percentage_in_polygon,
                    "aspect_ratio": aspect_ratio,
                    "speed": getattr(track, 'avg_speed', 0)
                }
            
            # 7. Aceptar como customer
            return "customer", {
                "area_percentage": area_percentage_in_polygon,
                "head_in_roi": self._point_in_polygon(cx, y1 + h * 0.15, mesa.poly),
                "torso_in_roi": self._point_in_polygon(cx, y1 + h * 0.4, mesa.poly),
                "aspect_ratio": aspect_ratio,
                "speed": getattr(track, 'avg_speed', 0),
                "yolo_segment_validation": True
            }
        
        # 8. Casos ambiguos -> Excluir
        return "excluded", {
            "reason": "ambiguous_classification",
            "area_percentage": area_percentage_in_polygon,
            "aspect_ratio": aspect_ratio,
            "speed": getattr(track, 'avg_speed', 0)
        }
    
    def _calculate_intersection_area(self, person_rect, mesa_poly):
        """Calcula área de intersección entre persona y mesa"""
        try:
            intersection = person_rect.intersection(mesa_poly)
            return intersection.area if intersection.is_valid else 0
        except:
            return 0
    
    def _apply_basic_filters(self, area_percentage, track, mesa, cx, cy):
        """Aplica filtros básicos de área, velocidad y banda vertical"""
        # Filtro de área mínima
        min_area_percentage = 0.08
        if area_percentage < min_area_percentage:
            return "excluded", {
                "reason": "insufficient_area", 
                "area_percentage": area_percentage,
                "required": min_area_percentage
            }
        
        # Filtro de velocidad (peatones muy rápidos)
        avg_speed = getattr(track, 'avg_speed', 0)
        if avg_speed > 80.0:
            return "excluded", {
                "reason": "moving_too_fast", 
                "speed": avg_speed,
                "area_percentage": area_percentage
            }
        
        # Banda vertical (si aplica)
        if mesa.y_band:
            y_min, y_max = mesa.y_band
            if not (y_min <= cy <= y_max):
                return "excluded", {"reason": "outside_y_band"}
        
        return None
    
    def _analyze_posture(self, track, aspect_ratio, area_percentage):
        """Analiza postura para determinar si es staff (de pie) o walking"""
        avg_speed = getattr(track, 'avg_speed', 0)
        
        # Detectar personas de pie con pies visibles
        is_standing_with_feet = False
        if self.detector and hasattr(self.detector, 'is_person_standing_with_feet_visible'):
            try:
                x1, y1, x2, y2 = track.xyxy
                bbox = [x1, y1, x2, y2]
                is_standing_with_feet = self.detector.is_person_standing_with_feet_visible(
                    self.current_frame, bbox
                )
            except Exception as e:
                print(f"Error YOLO Pose: {e}, usando fallback geométrico")
                is_standing_with_feet = aspect_ratio > 2.0
        else:
            is_standing_with_feet = aspect_ratio > 2.5
        
        # Detectar personas caminando
        is_walking = avg_speed > 12.0
        
        if is_standing_with_feet or is_walking:
            classification_type = "staff" if is_standing_with_feet else "excluded"
            return classification_type, {
                "reason": "person_standing_or_walking", 
                "aspect_ratio": aspect_ratio,
                "speed": avg_speed,
                "area_percentage": area_percentage,
                "yolo_pose_detected": is_standing_with_feet if self.detector else False
            }
        
        return None
    
    def _is_likely_seated(self, aspect_ratio, track):
        """Determina si una persona probablemente está sentada"""
        avg_speed = getattr(track, 'avg_speed', 0)
        return aspect_ratio <= 2.0 and avg_speed < 12.0
    
    def _validate_customer_segment(self, track, area_percentage):
        """Validación secundaria del segmento con YOLO"""
        if (self.detector and self.current_frame is not None and 
            hasattr(self.detector, 'validate_person_segment')):
            
            x1, y1, x2, y2 = track.xyxy
            bbox = [x1, y1, x2, y2]
            is_complete_person = self.detector.validate_person_segment(self.current_frame, bbox)
            
            if not is_complete_person:
                print(f"🚫 Rechazado track {track.id} - segmento no contiene persona completa")
                return "excluded", {
                    "reason": "segment_validation_failed",
                    "area_percentage": area_percentage,
                    "yolo_segment_validation": False
                }
        
        return None
    
    def _double_check_mesa_polygon(self, track, mesa):
        """Double check específico del polígono de mesa"""
        try:
            x1, y1, x2, y2 = track.xyxy
            
            # Obtener bounding box del polígono de mesa
            mesa_coords = mesa.polygon
            mesa_x_coords = [coord[0] for coord in mesa_coords]
            mesa_y_coords = [coord[1] for coord in mesa_coords]
            
            mesa_x_min = max(0, int(min(mesa_x_coords)))
            mesa_y_min = max(0, int(min(mesa_y_coords)))
            mesa_x_max = min(self.current_frame.shape[1], int(max(mesa_x_coords)))
            mesa_y_max = min(self.current_frame.shape[0], int(max(mesa_y_coords)))
            
            # Validar bounds
            if mesa_x_max <= mesa_x_min or mesa_y_max <= mesa_y_min:
                return {"valid": True, "reason": "invalid_mesa_bounds"}
            
            # Extraer ROI de mesa
            mesa_roi = self.current_frame[mesa_y_min:mesa_y_max, mesa_x_min:mesa_x_max]
            
            if mesa_roi.size == 0:
                return {"valid": True, "reason": "empty_mesa_roi"}
            
            # Ajustar coordenadas de persona al ROI
            person_x1_roi = max(0, x1 - mesa_x_min)
            person_y1_roi = max(0, y1 - mesa_y_min)
            person_x2_roi = min(mesa_roi.shape[1], x2 - mesa_x_min)
            person_y2_roi = min(mesa_roi.shape[0], y2 - mesa_y_min)
            
            # Verificar que persona esté en ROI
            if person_x2_roi <= person_x1_roi or person_y2_roi <= person_y1_roi:
                return {"valid": True, "reason": "person_outside_mesa_roi"}
            
            # Usar detector para análisis específico
            if (self.detector and hasattr(self.detector, '_has_head_or_torso_in_mesa_roi')):
                has_valid_body_parts = self.detector._has_head_or_torso_in_mesa_roi(
                    mesa_roi, 
                    person_x1_roi, person_y1_roi, 
                    person_x2_roi, person_y2_roi
                )
                
                if not has_valid_body_parts:
                    return {"valid": False, "reason": "only_feet_in_mesa_polygon"}
                
                return {"valid": True, "reason": "valid_body_parts_detected"}
            else:
                return {"valid": True, "reason": "no_pose_detector"}
                
        except Exception as e:
            print(f"Error en double check de mesa: {e}")
            return {"valid": True, "reason": f"error_in_check: {str(e)}"}
    
    def _point_in_polygon(self, x, y, polygon):
        """Verifica si un punto está dentro del polígono"""
        point = Point(x, y)
        return polygon.contains(point)
