#!/usr/bin/env python3
# test_simple.py - Test básico
print("🚀 Test básico iniciado")

try:
    from config import parse_args, get_detection_params
    print("✅ Config importado")
    
    from data_manager import DataManager
    print("✅ DataManager importado")
    
    from processor import VideoProcessor
    print("✅ VideoProcessor importado")
    
    config = parse_args()
    print("✅ Args parseados")
    
    # Test load mesas
    mesas = DataManager.load_mesas(config.rois_path)
    print(f"✅ Mesas cargadas: {len(mesas)}")
    
    # Test video info
    video_info = DataManager.get_video_info(config.video_path)
    print(f"✅ Video info: {video_info['width']}x{video_info['height']}")
    
    # Test detection params
    print("🔄 Obteniendo parámetros de detección...")
    logic_params = get_detection_params()
    print("✅ Parámetros obtenidos")
    
    # Test processor creation (this might be where it hangs)
    print("🔄 Creando processor...")
    processor = VideoProcessor(mesas, video_info, logic_params, config.conf_threshold)
    print("✅ Processor creado")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
