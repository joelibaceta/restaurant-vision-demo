# detector/person_detector.py — Detector principal refactorizado
from ultralytics import YOLO
import cv2
import numpy as np
from .pose_analyzer import PoseAnalyzer
from .segment_validator import SegmentValidator


class PersonDetector:
    """Detector principal de personas con validación avanzada"""
    
    def __init__(self, weights="yolov8n.pt", pose_weights="yolov8n-pose.pt", conf=0.5):
        self.model = YOLO(weights)
        self.conf = conf
        
        # Cargar modelo pose
        try:
            self.pose_model = YOLO(pose_weights)
            print(f"✅ YOLO Pose cargado: {pose_weights}")
        except Exception as e:
            self.pose_model = None
            print(f"⚠️  YOLO Pose no disponible: {e}")
            print("🔄 Usando filtro permisivo sin pose")
        
        # Inicializar componentes especializados
        self.pose_analyzer = PoseAnalyzer(self.pose_model, conf)
        self.segment_validator = SegmentValidator(self.model, self.pose_model, conf)

    def infer(self, frame):
        """Detección básica de personas en frame completo"""
        res = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)[0]
        dets = []
        if res and res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                dets.append({"xyxy": (float(x1), float(y1), float(x2), float(y2)), "conf": float(c)})
        return dets

    def infer_roi(self, frame, roi_polygon):
        """Detección de personas dentro de un ROI específico"""
        # Obtener bounding box del ROI
        roi_points = np.array(roi_polygon, dtype=np.int32)
        x_min, y_min = roi_points.min(axis=0)
        x_max, y_max = roi_points.max(axis=0)
        
        # Expandir ROI con padding
        padding = 50
        h, w = frame.shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Validar tamaño del ROI
        if x_max - x_min < 100 or y_max - y_min < 100:
            return []
        
        # Extraer y procesar ROI
        roi_frame = frame[y_min:y_max, x_min:x_max]
        
        try:
            roi_results = self.model.predict(roi_frame, conf=self.conf, classes=[0], verbose=False)[0]
            
            dets = []
            if roi_results and roi_results.boxes is not None and len(roi_results.boxes):
                xyxy = roi_results.boxes.xyxy.cpu().numpy()
                confs = roi_results.boxes.conf.cpu().numpy()
                
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    # Ajustar coordenadas al frame completo
                    abs_x1 = x1 + x_min
                    abs_y1 = y1 + y_min
                    abs_x2 = x2 + x_min
                    abs_y2 = y2 + y_min
                    
                    # Filtrar usando análisis de pose para ROI de mesa
                    if self.pose_analyzer.has_head_or_torso_in_roi(roi_frame, x1, y1, x2, y2):
                        dets.append({
                            "xyxy": (abs_x1, abs_y1, abs_x2, abs_y2), 
                            "conf": float(c)
                        })
            
            return dets
            
        except Exception as e:
            print(f"Error en detección ROI: {e}")
            return []

    def validate_person_segment(self, frame, bbox):
        """Validación secundaria de segmentos (delega al validador especializado)"""
        return self.segment_validator.validate_person_segment(frame, bbox)

    def is_person_standing_with_feet_visible(self, frame, bbox):
        """Determina si una persona está de pie con pies visibles (delega al analizador de pose)"""
        return self.pose_analyzer.is_person_standing_with_feet_visible(frame, bbox)

    def is_person_standing(self, frame, bbox):
        """Determina si una persona está de pie (versión simplificada)"""
        if self.pose_model is None:
            x1, y1, x2, y2 = bbox
            aspect_ratio = (y2 - y1) / max(x2 - x1, 1e-6)
            return aspect_ratio > 2.0
        
        # Para standing simple, usamos el análisis completo de pies visibles
        return self.pose_analyzer.is_person_standing_with_feet_visible(frame, bbox)

    # Métodos legacy para compatibilidad
    def _has_head_or_torso_in_mesa_roi(self, roi_frame, x1, y1, x2, y2):
        """Método legacy - delega al pose analyzer"""
        return self.pose_analyzer.has_head_or_torso_in_roi(roi_frame, x1, y1, x2, y2)
