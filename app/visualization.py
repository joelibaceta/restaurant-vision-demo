# visualization.py — Panel superior: color por estado y conteo de personas
import cv2
import numpy as np
from typing import List, Tuple, Optional

# ------------------------
# Utilidades UI
# ------------------------
def _panel(img, title: str, rows: list, topleft=(12,12)):
    """
    rows: lista de dicts con:
      {"text": "T1 — 2 personas", "color": (B,G,R)}
    """
    x0, y0 = topleft
    pad_x, pad_y = 8, 6  # Reducir padding
    line_h = 20  # Reducir altura de línea
    title_h = 22  # Reducir altura de título
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ancho máximo con fuentes más pequeñas
    (tw,_), _ = cv2.getTextSize(title, font, 0.5, 1)  # Título más pequeño
    max_w = tw
    for r in rows:
        (w,_), _ = cv2.getTextSize(r["text"], font, 0.4, 1)  # Texto más pequeño
        max_w = max(max_w, w)
    w_box = max_w + pad_x*2
    h_box = title_h + len(rows)*line_h + pad_y*2 + 4

    # sombra más sutil
    shadow = img.copy()
    cv2.rectangle(shadow, (x0+2, y0+2), (x0+w_box+2, y0+h_box+2), (0,0,0), -1)
    cv2.addWeighted(shadow, 0.15, img, 0.85, 0, img)

    # fondo más transparente
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+w_box, y0+h_box), (250,250,250), -1)
    cv2.rectangle(overlay, (x0, y0), (x0+w_box, y0+h_box), (200,200,200), 1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)  # Más transparente

    # título más pequeño
    cv2.putText(img, title, (x0+pad_x, y0+pad_y+title_h-6), font, 0.5, (40,40,40), 1, cv2.LINE_AA)

    # separador más sutil
    y = y0 + pad_y + title_h
    cv2.line(img, (x0+pad_x, y+1), (x0+w_box-pad_x, y+1), (180,180,180), 1, cv2.LINE_AA)
    y += 6

    # filas con texto más pequeño
    for r in rows:
        y += line_h
        # badge de color más pequeño
        cx = x0 + pad_x - 2
        cy = y - 10
        cv2.circle(img, (cx, cy), 4, r["color"], -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, (255,255,255), 1, lineType=cv2.LINE_AA)
        # texto más pequeño
        cv2.putText(img, r["text"], (x0+pad_x+8, y-6), font, 0.4, (30,30,30), 1, cv2.LINE_AA)

def render_people(frame, tracks=None, show_people=True, mesas=None):
    """Dibuja cajas de personas con información de estado.
    Solo muestra personas que están clasificadas como staff o clientes válidos."""
    if not (show_people and tracks and mesas):
        return frame
    
    # Crear conjuntos de IDs relevantes (staff + clientes en mesas)
    relevant_track_ids = set()
    all_staff_ids = set()
    
    for mesa in mesas:
        # Agregar staff
        if hasattr(mesa, 'staff_tracks'):
            all_staff_ids.update(mesa.staff_tracks)
            relevant_track_ids.update(mesa.staff_tracks)
        
        # Agregar clientes sentados/válidos
        if hasattr(mesa, 'tracks_in_area'):
            for track_id in mesa.tracks_in_area:
                if track_id not in all_staff_ids:  # No es staff
                    relevant_track_ids.add(track_id)
    
    for tr in tracks:
        # Solo mostrar tracks que son relevantes (staff o clientes válidos)
        if tr.id not in relevant_track_ids:
            continue
            
        x1,y1,x2,y2 = map(int, tr.xyxy)
        
        # Determinar si es staff
        is_staff = tr.id in all_staff_ids
        
        if is_staff:
            # Staff - siempre azul
            color = (255, 100, 0)  # Azul brillante
            status = f"STAFF {tr.speed:.1f}px/s"
            label_text = f"STAFF:{tr.id}"
        else:
            # Cliente - color basado en velocidad
            if hasattr(tr, 'speed'):
                if tr.speed < 12.0:  # Persona quieta/sentada
                    color = (0, 255, 0)  # Verde
                    status = f"QUIET {tr.speed:.1f}px/s"
                else:
                    color = (0, 165, 255)  # Naranja
                    status = f"MOVING {tr.speed:.1f}px/s"
            else:
                color = (255, 255, 255)  # Blanco por defecto
                status = "TRACKING"
            label_text = f"ID:{tr.id}"
        
        # Dibujar bbox con grosor diferente para staff
        thickness = 3 if is_staff else 2
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
        
        # ID del track con fondo diferente para staff
        label_bg_y = max(10, y1 - 10)
        label_width = 140 if is_staff else 120
        cv2.rectangle(frame, (x1, label_bg_y - 20), (x1 + label_width, label_bg_y), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, label_bg_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255) if is_staff else (0,0,0), 1)
        
        # Estado de velocidad
        cv2.putText(frame, status, (x1, y2 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return frame

def render(frame, mesas, tracks=None, show_people=True, capacities: Optional[dict]=None):
    """
    Muestra panel con estado detallado de las mesas:
      • círculo rojo si m.occupied == True, verde si False
      • texto: "<ID> — <n> personas" (y si pasas capacities, muestra sillas libres)
    """
    frame = render_people(frame, tracks, show_people=show_people, mesas=mesas)

    # Dibujar IDs de mesas en el centro de cada polígono
    for mesa in mesas:
        # Calcular centroide del polígono
        try:
            centroid = mesa.poly.centroid
            cx, cy = int(centroid.x), int(centroid.y)
            
            # Fondo semi-transparente sin borde
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), 25, (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # ID de la mesa (sin borde gris)
            text_size = cv2.getTextSize(mesa.id, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            cv2.putText(frame, mesa.id, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # NO dibujar contorno del polígono para evitar distracción visual
            
        except Exception as e:
            print(f"Error dibujando mesa {mesa.id}: {e}")

    rows = []
    mesas_sorted = sorted(mesas, key=lambda m: str(m.id))
    total_staff = 0
    
    for m in mesas_sorted:
        ppl = int(getattr(m, "people_seated", 0) or 0)
        staff_count = len(getattr(m, "staff_tracks", set()))
        total_staff += staff_count
        
        # Color más intuitivo: rojo para ocupada, verde para libre
        if m.occupied:
            color = (0, 0, 200)  # Rojo más intenso
            status_icon = ""  # Sin emoji, solo texto
        else:
            color = (0, 150, 0)  # Verde
            status_icon = ""  # Sin emoji, solo texto
            
        if capacities and m.id in capacities:
            cap = int(capacities[m.id])
            disp = max(cap - ppl, 0)
            if staff_count > 0:
                text = f"Mesa {m.id}: {ppl}/{cap} personas (disp: {disp}) + {staff_count} staff"
            else:
                text = f"Mesa {m.id}: {ppl}/{cap} personas (disp: {disp})"
        else:
            if staff_count > 0:
                text = f"Mesa {m.id}: {ppl} persona{'s' if ppl!=1 else ''} + {staff_count} staff"
            else:
                text = f"Mesa {m.id}: {ppl} persona{'s' if ppl!=1 else ''}"
            
        rows.append({"text": text, "color": color})

    # Información adicional en el panel
    total_occupied = sum(1 for m in mesas_sorted if m.occupied)
    total_people = sum(getattr(m, "people_seated", 0) for m in mesas_sorted)
    
    header_info = f"Ocupación: {total_occupied}/{len(mesas_sorted)} mesas | {total_people} clientes"
    _panel(frame, header_info, rows, topleft=(12,12))
    
    return frame