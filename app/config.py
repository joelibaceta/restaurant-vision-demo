# config.py — Configuración centralizada del sistema
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """Configuración principal de la aplicación"""
    video_path: str = "data/video.mov"
    rois_path: str = "data/rois.yaml"
    events_path: str = "data/events.csv"
    output_path: str = "data/out.mp4"
    conf_threshold: float = 0.5
    display: bool = True  # Cambiar a True por defecto
    save_video: bool = True


def parse_args() -> AppConfig:
    """Parsear argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Sistema de detección de ocupación de mesas")
    
    parser.add_argument("--video", default="data/video.mov", 
                       help="Ruta del video de entrada")
    parser.add_argument("--rois", default="data/rois.yaml", 
                       help="Archivo YAML con ROIs de las mesas")
    parser.add_argument("--conf", type=float, default=0.5, 
                       help="Umbral de confianza para detección")
    parser.add_argument("--save_video", default="data/out.mp4", 
                       help="Ruta del video de salida")
    parser.add_argument("--display", action="store_true", default=True,
                       help="Mostrar video en ventana (por defecto: True)")
    parser.add_argument("--no-display", dest="display", action="store_false",
                       help="No mostrar video en ventana")
    parser.add_argument("--events", default="data/events.csv", 
                       help="Archivo CSV para eventos")
    
    args = parser.parse_args()
    
    return AppConfig(
        video_path=args.video,
        rois_path=args.rois,
        events_path=args.events,
        output_path=args.save_video,
        conf_threshold=args.conf,
        display=args.display,
        save_video=bool(args.save_video)
    )


def get_detection_params():
    """Parámetros optimizados para detección"""
    from logic import LogicParams
    return LogicParams(
        conf_thr=0.5,
        min_bbox_frac=0.001,
        max_bbox_frac=0.09,
        v_thr_px_s=32.0,
        sit_seconds=2.0,
        hist_frames=6,
        ttl_lost=11.0,
        min_aspect_ratio=0.35,
        max_aspect_ratio=2.9,
        center_weight=0.75,
        min_stability_time=1.8,
        max_displacement_px=75.0
    )
