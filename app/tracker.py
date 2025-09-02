# tracker.py — Tracker liviano NN + velocidad (sin dependencias externas)
import time
import math

class Track:
    __slots__ = ("id","xyxy","cx","cy","last_t","speed","age","hits","misses","speed_history","avg_speed")
    def __init__(self, tid, xyxy, t):
        self.id = tid
        self.age = 0
        self.hits = 0
        self.misses = 0
        self.speed_history = []  # Historial de velocidades para suavizar
        self.avg_speed = 0.0
        self.update(xyxy, t, init=True)

    def update(self, xyxy, t, init=False):
        x1,y1,x2,y2 = xyxy
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        if init:
            self.speed = 0.0
        else:
            dt = max(1e-3, t - self.last_t)
            instant_speed = math.hypot(cx - self.cx, cy - self.cy) / dt
            
            # Mantener historial de velocidades (últimas 10 mediciones)
            self.speed_history.append(instant_speed)
            if len(self.speed_history) > 10:
                self.speed_history = self.speed_history[-10:]
            
            # Calcular velocidad promedio y actual
            self.avg_speed = sum(self.speed_history) / len(self.speed_history)
            self.speed = instant_speed
            
        self.xyxy = xyxy
        self.cx = cx
        self.cy = cy
        self.last_t = t
        self.age += 1
        self.hits += 1
        self.misses = 0

class SimpleTracker:
    def __init__(self, max_dist=80.0, max_misses=30):  # Más tolerante para oclusiones
        self.max_dist = max_dist
        self.max_misses = max_misses
        self._next_id = 1
        self.tracks = {}
        self.lost_tracks = {}  # Tracks perdidos que podrían reaparecer

    def update(self, detections):
        """detections: [{'xyxy':(x1,y1,x2,y2),'conf':float}] -> lista de Track activos"""
        t = time.time()
        # Asociación greedy por distancia de centro
        det_centers = [((d["xyxy"][0]+d["xyxy"][2])/2.0, (d["xyxy"][1]+d["xyxy"][3])/2.0) for d in detections]
        unmatched = set(range(len(detections)))
        
        # Intentar asociar a cada track activo
        for tid, tr in list(self.tracks.items()):
            # buscar la detección más cercana
            best_j, best_d = None, 1e9
            for j in unmatched:
                cx, cy = det_centers[j]
                d = math.hypot(cx - tr.cx, cy - tr.cy)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None and best_d <= self.max_dist:
                tr.update(detections[best_j]["xyxy"], t, init=False)
                unmatched.remove(best_j)
            else:
                tr.misses += 1
                # Si supera el límite, mover a tracks perdidos en lugar de eliminar
                if tr.misses > self.max_misses:
                    self.lost_tracks[tid] = tr
                    self.tracks.pop(tid, None)

        # Intentar reactivar tracks perdidos con detecciones no asignadas
        for j in list(unmatched):
            cx, cy = det_centers[j]
            best_tid, best_d = None, 1e9
            
            # Buscar en tracks perdidos
            for tid, tr in self.lost_tracks.items():
                # Usar distancia más flexible para reactivación
                d = math.hypot(cx - tr.cx, cy - tr.cy)
                if d < best_d and d <= self.max_dist * 2.0:  # Distancia más flexible
                    best_d, best_tid = d, tid
            
            if best_tid is not None:
                # Reactivar track perdido
                reactivated_track = self.lost_tracks.pop(best_tid)
                reactivated_track.update(detections[j]["xyxy"], t, init=False)
                reactivated_track.misses = 0  # Reset misses
                self.tracks[best_tid] = reactivated_track
                unmatched.remove(j)

        # Crear nuevos tracks para detecciones no asignadas
        for j in unmatched:
            tid = self._next_id; self._next_id += 1
            self.tracks[tid] = Track(tid, detections[j]["xyxy"], t)

        # Limpiar tracks perdidos muy antiguos (más de 60 segundos)
        current_time = time.time()
        for tid in list(self.lost_tracks.keys()):
            if current_time - self.lost_tracks[tid].last_t > 60.0:
                self.lost_tracks.pop(tid, None)

        return list(self.tracks.values())