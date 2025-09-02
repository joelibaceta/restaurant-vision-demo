import numpy as np

class SegmentValidator:
    def __init__(self, model, pose_model, conf_threshold=0.5):
        self.model = model
        self.pose_model = pose_model
        self.conf = conf_threshold

    def validate_person_segment(self, frame, bbox):
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                return False
            segment = self._extract_segment_with_padding(frame, x1, y1, x2, y2)
            if segment is None:
                return False
            best_detection = self._find_best_detection_in_segment(segment)
            if best_detection is None:
                return False
            return self._validate_detection_quality(segment, best_detection)
        except Exception as e:
            print(f"Error en validación de segmento: {e}")
            return False

    def _extract_segment_with_padding(self, frame, x1, y1, x2, y2):
        padding = 15
        h, w = frame.shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)
        segment = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if segment.size == 0 or segment.shape[0] < 40 or segment.shape[1] < 25:
            return None
        return segment

    def _find_best_detection_in_segment(self, segment):
        segment_results = self.model.predict(
            segment,
            conf=self.conf * 0.8,
            classes=[0],
            verbose=False
        )[0]
        if not segment_results or segment_results.boxes is None or len(segment_results.boxes) == 0:
            return None
        xyxy = segment_results.boxes.xyxy.cpu().numpy()
        confs = segment_results.boxes.conf.cpu().numpy()
        best_detection = None
        best_conf = 0
        for (sx1, sy1, sx2, sy2), conf in zip(xyxy, confs):
            seg_w, seg_h = sx2 - sx1, sy2 - sy1
            seg_area = seg_w * seg_h
            segment_area = segment.shape[0] * segment.shape[1]
            area_ratio = seg_area / segment_area
            if area_ratio > 0.2 and conf > best_conf:
                best_detection = (sx1, sy1, sx2, sy2)
                best_conf = conf
        return best_detection

    def _validate_detection_quality(self, segment, detection):
        sx1, sy1, sx2, sy2 = detection
        seg_w, seg_h = sx2 - sx1, sy2 - sy1
        seg_aspect_ratio = seg_h / max(seg_w, 1e-6)
        if seg_aspect_ratio < 0.8:
            return False
        if seg_h < 30 or seg_w < 15:
            return False
        if self.pose_model is not None:
            det_segment = segment[int(sy1):int(sy2), int(sx1):int(sx2)]
            if det_segment.size > 0:
                return self._validate_segment_has_head(det_segment)
        return True

    def _validate_segment_has_head(self, segment):
        try:
            pose_results = self.pose_model.predict(
                segment,
                conf=self.conf * 0.4,
                verbose=False
            )[0]
            if not pose_results or pose_results.keypoints is None or len(pose_results.keypoints) == 0:
                return False
            keypoints = pose_results.keypoints.xy.cpu().numpy()
            confidences = pose_results.keypoints.conf.cpu().numpy()
            for person_kpts, person_confs in zip(keypoints, confidences):
                head_kpts = [0, 1, 2, 3, 4]
                torso_kpts = [5, 6]
                lower_kpts = [11, 12, 13, 14, 15, 16]
                head_count = sum(1 for idx in head_kpts if person_confs[idx] > 0.3)
                torso_count = sum(1 for idx in torso_kpts if person_confs[idx] > 0.3)
                lower_count = sum(1 for idx in lower_kpts if person_confs[idx] > 0.3)
                has_head = head_count >= 1
                has_upper_torso = torso_count >= 1
                only_lower_body = lower_count > 0 and head_count == 0 and torso_count == 0
                if only_lower_body:
                    return False
                if has_head or has_upper_torso:
                    return True
            return False
        except Exception:
            return False
