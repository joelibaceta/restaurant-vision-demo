# main.py — Sistema de detección de ocupación actualizado

import cv2
import yaml
import time
import pandas as pd
from pathlib import Path

# Importar desde la nueva estructura modular
from detector import PersonDetector  # Ahora es un wrapper
from logic import OccupancyEngine, Mesa, LogicParams  # Ahora son wrappers
from tracker import PersonTracker
from visualization import Visualizer

def load_mesas(filepath):
    """Cargar configuración de mesas desde archivo YAML"""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    mesas = []
    for mesa_data in data['mesas']:
        mesa = Mesa(
            id=mesa_data['id'],
            polygon=mesa_data['polygon'],
            iop_thr=mesa_data.get('iop_thr', 0.12),
            y_band=mesa_data.get('y_band')
        )
        mesas.append(mesa)
    
    return mesas

def main():
    """Función principal del sistema"""
    
    # Configuración
    config = {
        'video_path': 'data/video.mov',
        'mesas_path': 'data/rois.yaml',
        'events_path': 'data/events.csv',
        'output_path': 'data/out.mp4',
        'display': True,
        'save_video': True
    }
    
    # Inicializar componentes
    print("🚀 Iniciando sistema de detección de ocupación...")
    
    # Cargar mesas
    mesas = load_mesas(config['mesas_path'])
    print(f"📋 Cargadas {len(mesas)} mesas")
    
    # Abrir video
    cap = cv2.VideoCapture(config['video_path'])
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video {config['video_path']}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {frame_w}x{frame_h} @ {fps:.1f}fps, {total_frames} frames")
    
    # Inicializar detector con parámetros optimizados
    detector = PersonDetector(
        conf_threshold=0.5,
        device='cpu'  # Cambiar a 'cuda' si tienes GPU
    )
    
    # Parámetros optimizados de lógica
    logic_params = LogicParams(
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
    
    # Inicializar tracker y motor de ocupación
    tracker = PersonTracker()
    occupancy_engine = OccupancyEngine(
        mesas=mesas,
        frame_size=(frame_h, frame_w),
        params=logic_params,
        detector=detector  # Pasar detector para validación de segmentos
    )
    
    # Inicializar visualizador
    visualizer = Visualizer(mesas)
    
    # Configurar salida de video si es necesario
    out_writer = None
    if config['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(config['output_path'], fourcc, fps, (frame_w, frame_h))
    
    # Estadísticas
    events = []
    frame_count = 0
    start_time = time.time()
    
    print("▶️  Iniciando procesamiento...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps
            
            # 1. Detectar personas
            detections = detector.detect(frame)
            
            # 2. Tracking
            tracks = tracker.update(detections)
            
            # 3. Análisis de ocupación
            occupancy_engine.step(tracks, frame)
            
            # 4. Registrar eventos
            for mesa in mesas:
                events.append({
                    'frame': frame_count,
                    'time': current_time,
                    'mesa_id': mesa.id,
                    'occupied': mesa.occupied,
                    'people_seated': mesa.people_seated
                })
            
            # 5. Visualización
            vis_frame = visualizer.draw(frame, tracks, mesas, detections)
            
            # 6. Mostrar progreso
            if frame_count % 60 == 0:  # Cada 2 segundos aprox
                progress = frame_count / total_frames * 100
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"🎬 Frame {frame_count}/{total_frames} ({progress:.1f}%) - {fps_actual:.1f} fps")
            
            # 7. Guardar frame
            if out_writer:
                out_writer.write(vis_frame)
            
            # 8. Mostrar en ventana (opcional)
            if config['display']:
                # Redimensionar para visualización si es muy grande
                display_frame = vis_frame
                if frame_w > 1280:
                    scale = 1280 / frame_w
                    new_w = int(frame_w * scale)
                    new_h = int(frame_h * scale)
                    display_frame = cv2.resize(vis_frame, (new_w, new_h))
                
                cv2.imshow('Restaurant Vision Demo', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Detenido por usuario")
                    break
    
    except KeyboardInterrupt:
        print("🛑 Interrumpido por usuario")
    
    finally:
        # Limpiar recursos
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        
        # Guardar eventos
        df_events = pd.DataFrame(events)
        df_events.to_csv(config['events_path'], index=False)
        
        # Estadísticas finales
        elapsed = time.time() - start_time
        fps_avg = frame_count / elapsed
        print(f"\n📊 Procesamiento completado:")
        print(f"   • Frames procesados: {frame_count}")
        print(f"   • Tiempo total: {elapsed:.1f}s")
        print(f"   • FPS promedio: {fps_avg:.1f}")
        print(f"   • Eventos guardados: {config['events_path']}")
        if config['save_video']:
            print(f"   • Video guardado: {config['output_path']}")

if __name__ == "__main__":
    main()
import argparse, csv, datetime, os, yaml, cv2
from shapely.geometry import Polygon
from detector import PersonDetector
from tracker import SimpleTracker
from logic import Mesa, OccupancyEngine, LogicParams
from visualization import render

def load_mesas(rois_path, iop_near=0.02, iop_far=0.04):  # Umbrales muy bajos para ser más inclusivo
    with open(rois_path, "r") as f:
        data = yaml.safe_load(f)
    mesas = []
    capacities = {}
    areas = []
    for t in data["tables"]:
        poly = Polygon([(float(x),float(y)) for x,y in t["polygon"]])
        areas.append(poly.area)
    if not areas: return [], {}
    area_med = sorted(areas)[len(areas)//2]
    for t in data["tables"]:
        poly_pts = [(int(x),int(y)) for x,y in t["polygon"]]
        poly_area = Polygon([(float(x),float(y)) for x,y in poly_pts]).area
        mesa = Mesa(mesa_id := t["id"], polygon=poly_pts,
                    iop_thr=(iop_far if poly_area < area_med else iop_near))
        mesas.append(mesa)
        if "capacity" in t:
            capacities[mesa_id] = int(t["capacity"])
    return mesas, capacities

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/demo_restaurant.mp4")
    ap.add_argument("--rois", default="data/rois.yaml")
    ap.add_argument("--out_csv", default="data/events.csv")
    ap.add_argument("--save_video", default="")  # e.g., data/out.mp4
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    mesas, capacities = load_mesas(args.rois)
    if not mesas:
        raise SystemExit("No hay mesas en el YAML.")

    detector = PersonDetector(conf=args.conf, pose_weights="yolov8n-pose.pt")
    tracker = SimpleTracker(max_dist=100.0, max_misses=30)  # Más tolerante para personas sentadas
    
    # Parámetros optimizados para detección de personas sentadas
    from logic import LogicParams
    logic_params = LogicParams(
        min_bbox_frac=0.0001,      # Muy permisivo para detecciones pequeñas
        max_bbox_frac=0.2,         # Más permisivo para detecciones grandes
        v_thr_px_s=40.0,          # Velocidad máxima MÁS ALTA para movimiento sentado
        sit_seconds=1.5,          # Tiempo más corto para confirmar sentado
        hist_frames=3,            # Menos frames para histéresis más rápida
        ttl_lost=30.0,            # MÁS tiempo para mantener track perdido (oclusiones)
        min_aspect_ratio=0.1,     # Ratio mínimo muy permisivo
        max_aspect_ratio=5.0,     # Ratio máximo muy permisivo
        min_stability_time=0.5    # Tiempo mínimo de estabilidad MUY CORTO
    )
    
    engine = OccupancyEngine(mesas, frame_size=(h,w), params=logic_params, detector=detector)

    # writer de video opcional
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (w,h))

    # CSV
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    if not os.path.exists(args.out_csv):
        with open(args.out_csv, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","table_id","occupied","people_seated"])

    last_csv = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break

        # DETECCIÓN HÍBRIDA: Global + ROI específico por mesa
        all_dets = detector.infer(frame)
        
        # Detección complementaria por ROI de mesa
        roi_dets = []
        roi_raw_count = 0
        for mesa in mesas:
            mesa_dets = detector.infer_roi(frame, mesa.polygon)
            roi_dets.extend(mesa_dets)
        
        # Combinar detecciones evitando duplicados
        combined_dets = engine.combine_detections(all_dets, roi_dets)

        tracks = tracker.update(combined_dets)

        engine.step(tracks, frame)

        frame = render(frame, mesas, tracks, show_people=True, capacities=capacities)
        cv2.imshow("Remi Demo", frame)
        if writer: writer.write(frame)

        # Log CSV ~1 Hz
        now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        if (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0) - last_csv >= 1.0:
            with open(args.out_csv, "a", newline="") as f:
                cw = csv.writer(f)
                for m in mesas:
                    cw.writerow([now, m.id, int(m.occupied), m.people_seated])
            last_csv = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()