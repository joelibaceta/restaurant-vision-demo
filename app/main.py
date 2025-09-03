# main.py — Orquestador principal del sistema de detección de ocupación

import cv2
import sys
from pathlib import Path

# Importar módulos del sistema
from config import parse_args, get_detection_params
from data_manager import DataManager
from processor import VideoProcessor


def main():
    """🎭 Orquestador principal - Coordina todos los componentes del sistema"""
    
    print("DEBUG: Iniciando main()")
    
    try:
        # 📋 1. Configuración
        print("DEBUG: Parseando argumentos...")
        print("🚀 Iniciando sistema de detección de ocupación...")
        config = parse_args()
        print(f"DEBUG: Config obtenido: {config}")
        
        # 📊 2. Cargar datos
        print("📂 Cargando configuración...")
        mesas = DataManager.load_mesas(config.rois_path)
        print(f"📋 Cargadas {len(mesas)} mesas")
        
        video_info = DataManager.get_video_info(config.video_path)
        print(f"📹 Video: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps, {video_info['total_frames']} frames")
        
        # ⚙️ 3. Inicializar procesador
        print("⚙️ Configurando procesador...")
        logic_params = get_detection_params()
        processor = VideoProcessor(mesas, video_info, logic_params, config.conf_threshold)
        
        # 📹 4. Configurar salida de video
        out_writer = None
        if config.save_video:
            out_writer = DataManager.setup_video_writer(
                config.output_path, 
                video_info['fps'], 
                video_info['width'], 
                video_info['height']
            )
        
        # 🎬 5. Procesar video
        print("▶️ Iniciando procesamiento...")
        cap = video_info['cap']
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            vis_frame, tracks = processor.process_frame(frame)
            
            # Mostrar progreso
            if processor.should_show_progress():
                progress = processor.get_progress_info()
                print(f"🎬 Frame {progress['frame']}/{progress['total_frames']} ({progress['progress']:.1f}%) - {progress['fps']:.1f} fps")
            
            # Guardar frame
            if out_writer:
                out_writer.write(vis_frame)
            
            # Mostrar en ventana (opcional)
            if config.display:
                display_frame = _prepare_display_frame(vis_frame, video_info['width'])
                cv2.imshow('Restaurant Vision Demo', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Detenido por usuario")
                    break
        
        # 💾 6. Guardar resultados
        print("💾 Guardando resultados...")
        DataManager.save_events(processor.events, config.events_path)
        
        # 📊 7. Mostrar estadísticas finales
        _show_final_stats(processor, config)
        
    except KeyboardInterrupt:
        print("🛑 Interrumpido por usuario")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        # 🧹 Limpiar recursos
        _cleanup_resources(video_info.get('cap'), out_writer)


def _prepare_display_frame(frame, original_width):
    """Preparar frame para visualización redimensionando si es necesario"""
    if original_width > 1280:
        scale = 1280 / original_width
        new_w = int(original_width * scale)
        new_h = int(frame.shape[0] * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame


def _show_final_stats(processor, config):
    """Mostrar estadísticas finales del procesamiento"""
    stats = processor.get_final_stats()
    print(f"\n📊 Procesamiento completado:")
    print(f"   • Frames procesados: {stats['frames_processed']}")
    print(f"   • Tiempo total: {stats['total_time']:.1f}s")
    print(f"   • FPS promedio: {stats['avg_fps']:.1f}")
    print(f"   • Eventos guardados: {config.events_path}")
    if config.save_video:
        print(f"   • Video guardado: {config.output_path}")


def _cleanup_resources(cap, out_writer):
    """Limpiar recursos del sistema"""
    if cap:
        cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()