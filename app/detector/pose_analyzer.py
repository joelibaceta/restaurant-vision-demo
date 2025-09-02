# detector/pose_analyzer.py — Análisis con YOLO Pose
import numpy as np


class PoseAnalyzer:
    """Maneja todo el análisis de pose usando YOLO Pose"""
    
    def __init__(self, pose_model, conf_threshold=0.5):
        self.pose_model = pose_model
        self.conf = conf_threshold
    
    def has_head_or_torso_in_roi(self, roi_frame, x1, y1, x2, y2):
        """
        Filtro ESTRICTO para ROI de mesa: NO contar solo pies dentro del polígono.
        Solo acepta detecciones que incluyan cabeza/torso/brazos, rechaza solo piernas/pies.
        """
        if self.pose_model is None:
            return self._geometric_head_torso_check(roi_frame, x1, y1, x2, y2)
        
        try:
            y1_int, y2_int = max(0, int(y1)), min(roi_frame.shape[0], int(y2))
            x1_int, x2_int = max(0, int(x1)), min(roi_frame.shape[1], int(x2))
            
            det_height = y2_int - y1_int
            det_width = x2_int - x1_int
            roi_height, roi_width = roi_frame.shape[:2]
            
            # Filtros geométricos previos
            if not self._passes_geometric_filters(det_height, det_width, roi_height, y1_int):
                return False
            
            det_frame = roi_frame[y1_int:y2_int, x1_int:x2_int]
            if det_frame.size == 0:
                return False

            # Ejecutar YOLO Pose
            pose_results = self.pose_model.predict(det_frame, conf=self.conf * 0.3, verbose=False)[0]
            
            if pose_results and pose_results.keypoints is not None and len(pose_results.keypoints):
                return self._analyze_keypoints_for_mesa(pose_results)
            
            return False
            
        except Exception:
            return self._geometric_head_torso_check(roi_frame, x1, y1, x2, y2)

    def _passes_geometric_filters(self, det_height, det_width, roi_height, y1_int):
        """Filtros geométricos para rechazar formas típicas de solo pies/piernas"""
        aspect_ratio = det_height / max(det_width, 1e-6)
        relative_top = y1_int / roi_height
        relative_height = det_height / roi_height
        
        # RECHAZAR si está muy abajo en el ROI
        if relative_top > 0.7:
            return False
            
        # RECHAZAR si es muy pequeño
        if relative_height < 0.3:
            return False
            
        # RECHAZAR formas anchas y bajas típicas de pies
        if aspect_ratio < 1.2 and relative_top > 0.5:
            return False
            
        # RECHAZAR detecciones muy pequeñas en la parte inferior
        if det_height < 60 and relative_top > 0.6:
            return False
            
        # RECHAZAR detecciones muy bajas
        if det_height < 80:
            return False
        
        # Filtro básico de tamaño
        if det_height < 30 or det_width < 15:
            return False
            
        return True

    def _analyze_keypoints_for_mesa(self, pose_results):
        """Analiza keypoints específicamente para detección en mesa"""
        keypoints = pose_results.keypoints.xy.cpu().numpy()
        confidences = pose_results.keypoints.conf.cpu().numpy()
        
        for person_kpts, person_confs in zip(keypoints, confidences):
            # Categorizar keypoints por región del cuerpo
            head_kpts = [0, 1, 2, 3, 4]  # nariz, ojos, orejas
            torso_kpts = [5, 6, 7, 8, 9, 10]  # hombros, codos, muñecas
            lower_kpts = [11, 12, 13, 14, 15, 16]  # caderas, rodillas, tobillos
            
            head_count = sum(1 for idx in head_kpts if person_confs[idx] > 0.3)
            torso_count = sum(1 for idx in torso_kpts if person_confs[idx] > 0.3)
            lower_count = sum(1 for idx in lower_kpts if person_confs[idx] > 0.3)
            
            # RECHAZAR si SOLO detecta parte inferior
            if lower_count >= 1 and head_count == 0 and torso_count == 0:
                return False
            
            # RECHAZAR si detecta principalmente tobillos/pies sin contexto superior
            ankle_kpts = [15, 16]
            ankle_count = sum(1 for idx in ankle_kpts if person_confs[idx] > 0.3)
            if ankle_count >= 1 and head_count == 0 and torso_count <= 1:
                return False
                
            # ACEPTAR con evidencia clara de cabeza O torso
            if head_count >= 2 or torso_count >= 2:
                return True
                
        return False

    def _geometric_head_torso_check(self, roi_frame, x1, y1, x2, y2):
        """Filtro geométrico MUY ESTRICTO contra piernas/pies"""
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        roi_height, roi_width = roi_frame.shape[:2]
        
        aspect_ratio = bbox_height / max(bbox_width, 1e-6)
        if aspect_ratio < 1.5:
            return False
        
        relative_top = y1 / roi_height
        if relative_top > 0.5:
            return False
        
        height_ratio = bbox_height / roi_height
        if height_ratio < 0.4:
            return False
        
        if relative_top > 0.3 and aspect_ratio < 2.0:
            return False
        
        return True

    def is_person_standing_with_feet_visible(self, frame, bbox):
        """Determina si una persona está de pie Y tiene pies visibles"""
        if self.pose_model is None:
            x1, y1, x2, y2 = bbox
            aspect_ratio = (y2 - y1) / max(x2 - x1, 1e-6)
            return aspect_ratio > 2.0
        
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Expandir para mejor análisis
            padding = 10
            x1_exp = max(0, int(x1) - padding)
            y1_exp = max(0, int(y1) - padding)
            x2_exp = min(w, int(x2) + padding)
            y2_exp = min(h, int(y2) + padding)
            
            person_frame = frame[y1_exp:y2_exp, x1_exp:x2_exp]
            if person_frame.size == 0:
                return False
            
            pose_results = self.pose_model.predict(person_frame, conf=self.conf * 0.4, verbose=False)[0]
            
            if pose_results and pose_results.keypoints is not None and len(pose_results.keypoints):
                return self._analyze_standing_posture(pose_results, person_frame)
            
            return False
            
        except Exception:
            aspect_ratio = (y2 - y1) / max(x2 - x1, 1e-6)
            return aspect_ratio > 2.8

    def _analyze_standing_posture(self, pose_results, person_frame):
        """Analiza si la postura indica una persona de pie completa"""
        keypoints = pose_results.keypoints.xy.cpu().numpy()
        confidences = pose_results.keypoints.conf.cpu().numpy()
        
        for person_kpts, person_confs in zip(keypoints, confidences):
            # Keypoints específicos
            ankle_kpts = [15, 16]  # tobillos
            knee_kpts = [13, 14]   # rodillas
            hip_kpts = [11, 12]    # caderas
            head_kpts = [0, 1, 2, 3, 4]  # cabeza
            shoulder_kpts = [5, 6]  # hombros
            
            ankles_visible = sum(1 for idx in ankle_kpts if person_confs[idx] > 0.4)
            knees_visible = sum(1 for idx in knee_kpts if person_confs[idx] > 0.4)
            hips_visible = sum(1 for idx in hip_kpts if person_confs[idx] > 0.4)
            head_visible = sum(1 for idx in head_kpts if person_confs[idx] > 0.4)
            shoulders_visible = sum(1 for idx in shoulder_kpts if person_confs[idx] > 0.4)
            
            # Debe tener estructura completa de persona DE PIE
            has_full_head = head_visible >= 2
            has_torso = shoulders_visible >= 1 and hips_visible >= 1
            has_complete_legs = ankles_visible >= 1 and knees_visible >= 1
            
            if has_full_head and has_torso and has_complete_legs:
                return self._verify_vertical_alignment(person_kpts, person_confs, person_frame)
        
        return False

    def _verify_vertical_alignment(self, person_kpts, person_confs, person_frame):
        """Verifica alineación vertical para confirmar postura erguida"""
        visible_keypoints = {}
        for idx in range(17):
            if person_confs[idx] > 0.4:
                visible_keypoints[idx] = person_kpts[idx]
        
        if len(visible_keypoints) >= 5:
            # Obtener posiciones Y
            head_kpts = [0, 1, 2, 3, 4]
            shoulder_kpts = [5, 6]
            hip_kpts = [11, 12]
            knee_kpts = [13, 14]
            ankle_kpts = [15, 16]
            
            head_y = min([visible_keypoints[i][1] for i in head_kpts if i in visible_keypoints], default=float('inf'))
            shoulder_y = min([visible_keypoints[i][1] for i in shoulder_kpts if i in visible_keypoints], default=float('inf'))
            hip_y = min([visible_keypoints[i][1] for i in hip_kpts if i in visible_keypoints], default=float('inf'))
            knee_y = min([visible_keypoints[i][1] for i in knee_kpts if i in visible_keypoints], default=float('inf'))
            ankle_y = min([visible_keypoints[i][1] for i in ankle_kpts if i in visible_keypoints], default=float('inf'))
            
            # Verificar secuencia vertical
            vertical_sequence = head_y < shoulder_y < hip_y < knee_y < ankle_y
            
            # Verificar altura total
            total_height = ankle_y - head_y
            frame_height = person_frame.shape[0]
            height_ratio = total_height / frame_height
            
            return vertical_sequence and height_ratio > 0.7
        
        return False
