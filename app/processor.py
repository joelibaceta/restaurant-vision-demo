# processor.py — Motor de procesamiento principal
import time
from typing import List, Dict, Any, Optional
from detector import PersonDetector
from tracker import SimpleTracker as PersonTracker
from logic import OccupancyEngine, Mesa, LogicParams
from visualization import render


class VideoProcessor:
    """Procesador principal del video"""
    
    def __init__(self, mesas: List[Mesa], video_info: Dict, logic_params: LogicParams, conf_threshold: float = 0.5):
        self.mesas = mesas
        self.video_info = video_info
        
        # Inicializar componentes
        self.detector = PersonDetector(conf=conf_threshold)
        self.tracker = PersonTracker()
        self.occupancy_engine = OccupancyEngine(
            mesas=mesas,
            frame_size=(video_info['height'], video_info['width']),
            params=logic_params,
            detector=self.detector
        )
        
        # Estadísticas
        self.frame_count = 0
        self.start_time = time.time()
        self.events = []
    
    def process_frame(self, frame) -> tuple:
        """Procesar un frame individual"""
        self.frame_count += 1
        current_time = self.frame_count / self.video_info['fps']
        
        # Pipeline de procesamiento
        detections = self.detector.infer(frame)
        tracks = self.tracker.update(detections)
        self.occupancy_engine.step(tracks, frame)
        
        # Registrar eventos
        for mesa in self.mesas:
            self.events.append({
                'frame': self.frame_count,
                'time': current_time,
                'mesa_id': mesa.id,
                'occupied': mesa.occupied,
                'people_seated': mesa.people_seated
            })
        
        # Generar frame visualizado
        vis_frame = render(frame, self.mesas, tracks)
        
        return vis_frame, tracks
    
    def should_show_progress(self, interval: int = 60) -> bool:
        """Determinar si mostrar progreso"""
        return self.frame_count % interval == 0
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Obtener información de progreso"""
        progress = self.frame_count / self.video_info['total_frames'] * 100
        elapsed = time.time() - self.start_time
        fps_actual = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'frame': self.frame_count,
            'total_frames': self.video_info['total_frames'],
            'progress': progress,
            'fps': fps_actual,
            'elapsed': elapsed
        }
    
    def get_final_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas finales"""
        elapsed = time.time() - self.start_time
        fps_avg = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'total_time': elapsed,
            'avg_fps': fps_avg,
            'events_count': len(self.events)
        }
