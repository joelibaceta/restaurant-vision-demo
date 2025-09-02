# data_manager.py — Manejo de datos de entrada y salida
import yaml
import pandas as pd
import cv2
from typing import List, Dict, Any
from logic import Mesa


class DataManager:
    """Gestiona la carga y guardado de datos del sistema"""
    
    @staticmethod
    def load_mesas(filepath: str) -> List[Mesa]:
        """Cargar configuración de mesas desde archivo YAML"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        mesas = []
        # El archivo puede usar 'tables' o 'mesas'
        tables_key = 'tables' if 'tables' in data else 'mesas'
        
        for mesa_data in data[tables_key]:
            mesa = Mesa(
                id=mesa_data['id'],
                polygon=mesa_data['polygon'],
                iop_thr=mesa_data.get('iop_thr', 0.12),
                y_band=mesa_data.get('y_band')
            )
            mesas.append(mesa)
        
        return mesas
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """Obtener información del video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'cap': cap
        }
        
        return info
    
    @staticmethod
    def save_events(events: List[Dict], filepath: str) -> None:
        """Guardar eventos en archivo CSV"""
        df_events = pd.DataFrame(events)
        df_events.to_csv(filepath, index=False)
    
    @staticmethod
    def setup_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
        """Configurar el escritor de video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
