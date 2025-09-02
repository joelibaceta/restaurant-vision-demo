# roi_editor.py
import argparse
import os
import sys
import yaml
import cv2
import numpy as np
from typing import List, Tuple, Optional

# ------------------------------------------------------------
# ROI Editor — dibuja polígonos (mesas) y guarda rois.yaml
#
# Controles:
#  - Click izquierdo: añadir punto al polígono actual
#  - Click derecho o Enter: cerrar polígono
#  - u: deshacer último punto
#  - d: borrar último polígono
#  - s: guardar YAML
#  - q o Esc: salir
#
# Al cerrar un polígono, puedes escribir un ID (ej. "T1") y pulsar Enter.
# Si lo dejas vacío, se asigna automáticamente: T1, T2, ...
# ------------------------------------------------------------

def parse_timecode(tc: str) -> int:
    """Convierte 'HH:MM:SS' o 'MM:SS' a segundos."""
    parts = [int(p) for p in tc.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        raise ValueError("timestamp debe ser MM:SS o HH:MM:SS")
    return h * 3600 + m * 60 + s

def load_frame_from_source(source: str,
                           frame_idx: Optional[int] = None,
                           timestamp_s: Optional[int] = None) -> np.ndarray:
    """Abre imagen/video/cámara y devuelve un frame."""
    cap = None
    if os.path.exists(source):
        cap = cv2.VideoCapture(source)
    else:
        # Si no existe como archivo, intentamos índice de cámara
        try:
            cam_idx = int(source)
            cap = cv2.VideoCapture(cam_idx)
        except Exception:
            raise FileNotFoundError(f"No se encontró la fuente: {source}")

    if not cap or not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")

    if frame_idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    elif timestamp_s is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_s) * 1000.0)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("No se pudo leer un frame de la fuente.")
    return frame

class ROIEditor:
    def __init__(self, img: np.ndarray, out_path: str = "rois.yaml",
                 preload_polys: Optional[List[List[Tuple[int,int]]]] = None,
                 preload_ids: Optional[List[str]] = None):
        self.out_path = out_path
        self.base = img.copy()
        self.draw_img = img.copy()

        # Polígono actual en edición
        self.curr_pts: List[Tuple[int,int]] = []

        # Lista de polígonos [(pts, id), ...]
        self.polys: List[Tuple[List[Tuple[int,int]], str]] = []

        if preload_polys:
            for i, poly in enumerate(preload_polys):
                pid = preload_ids[i] if (preload_ids and i < len(preload_ids)) else f"T{i+1}"
                self.polys.append((poly, pid))

        # Para capturar texto cuando cerramos un polígono (ID de mesa)
        self.typing_id = False
        self.input_buffer = ""

        self.window = "ROI Editor — Left:add | Right/Enter:close | u:undo | d:del poly | s:save | q:quit"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)

    # --------- Interacción de mouse ----------
    def on_mouse(self, event, x, y, flags, param):
        if self.typing_id:
            return  # mientras se escribe un ID, ignoramos clicks

        if event == cv2.EVENT_LBUTTONDOWN:
            self.curr_pts.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.close_polygon()

    # --------- Lógica de cierre + ID ----------
    def close_polygon(self):
        if len(self.curr_pts) >= 3:
            # activa modo escritura de ID
            self.typing_id = True
            self.input_buffer = ""

    def finish_current_polygon(self, pid: Optional[str] = None):
        if len(self.curr_pts) >= 3:
            if not pid or not pid.strip():
                pid = f"T{len(self.polys)+1}"
            self.polys.append((self.curr_pts.copy(), pid.strip()))
            self.curr_pts.clear()
        self.typing_id = False
        self.input_buffer = ""

    # --------- Dibujo ----------
    def draw(self):
        self.draw_img = self.base.copy()

        # dibujar polígonos existentes
        for i, (poly, pid) in enumerate(self.polys):
            pts = np.array(poly, dtype=np.int32)
            # relleno suave
            overlay = self.draw_img.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            cv2.addWeighted(overlay, 0.25, self.draw_img, 0.75, 0, self.draw_img)
            # contorno
            cv2.polylines(self.draw_img, [pts], True, (0, 100, 0), 2, lineType=cv2.LINE_AA)
            # etiqueta
            top_pt = min(poly, key=lambda p: p[1])
            cv2.putText(self.draw_img, pid, (top_pt[0], max(20, top_pt[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15,15,15), 2, cv2.LINE_AA)

        # polígono en edición
        if self.curr_pts:
            pts = np.array(self.curr_pts, dtype=np.int32)
            for p in self.curr_pts:
                cv2.circle(self.draw_img, p, 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.polylines(self.draw_img, [pts], False, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # overlay de input si estamos solicitando ID
        if self.typing_id:
            msg = f"ID de mesa: {self.input_buffer}_"
            self._draw_centered_banner(msg)

    def _draw_centered_banner(self, text: str):
        h, w = self.draw_img.shape[:2]
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x = (w - tw) // 2
        y = 40
        cv2.rectangle(self.draw_img, (x-10, y-th-10), (x+tw+10, y+10), (255,255,255), -1)
        cv2.putText(self.draw_img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,20), 2, cv2.LINE_AA)

    # --------- Persistencia ----------
    def save_yaml(self):
        data = {
            "tables": [
                {"id": pid, "polygon": [[int(x), int(y)] for (x, y) in poly]}
                for (poly, pid) in self.polys
            ]
        }
        with open(self.out_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"[guardado] {self.out_path}")

    # --------- Loop principal ----------
    def run(self):
        while True:
            self.draw()
            cv2.imshow(self.window, self.draw_img)
            key = cv2.waitKey(20) & 0xFF

            # Captura de texto para el ID del polígono
            if self.typing_id:
                if key in (13, 10):       # Enter
                    self.finish_current_polygon(self.input_buffer)
                elif key in (27, ord('q')):  # Esc/Q cancela el naming (usa ID auto)
                    self.finish_current_polygon(None)
                elif key in (8, 127):     # Backspace/Delete
                    self.input_buffer = self.input_buffer[:-1]
                elif key != 255:          # otros visibles
                    ch = chr(key)
                    # Acepta letras/números/guiones/guión bajo/espacios
                    if ch.isalnum() or ch in "-_ ":
                        self.input_buffer += ch
                continue  # mientras tipeamos, ignorar el resto

            # Teclas normales
            if key in (27, ord('q')):  # Esc o q
                break
            elif key == ord('u'):
                if self.curr_pts:
                    self.curr_pts.pop()
            elif key == ord('d'):
                if self.polys:
                    self.polys.pop()
            elif key == ord('s'):
                # cerrar polígono abierto (si tiene >=3 puntos) antes de guardar
                if len(self.curr_pts) >= 3:
                    self.close_polygon()
                    # si entra en modo typing, seguir al próximo loop para completar ID
                    if not self.typing_id:
                        self.finish_current_polygon(None)
                self.save_yaml()
            elif key in (13, 10):  # Enter: cerrar polígono
                self.close_polygon()

        cv2.destroyAllWindows()
        # auto-guardar si hay polígonos y aún no existe archivo
        if self.polys and not os.path.exists(self.out_path):
            self.save_yaml()

def load_preload_yaml(path: str):
    if not path or not os.path.exists(path):
        return None, None
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    polys, ids = [], []
    for t in data.get("tables", []):
        poly = [(int(x), int(y)) for x, y in t.get("polygon", [])]
        pid = str(t.get("id", f"T{len(polys)+1}"))
        polys.append(poly)
        ids.append(pid)
    print(f"[cargado] {path} ({len(polys)} polígonos)")
    return polys, ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Ruta a video (.mp4), imagen (.jpg/.png) o índice de cámara (0/1/2)")
    ap.add_argument("--frame", type=int, default=None, help="Índice de frame (si es video)")
    ap.add_argument("--timestamp", type=str, default=None,
                    help="Timestamp MM:SS o HH:MM:SS para capturar frame de video")
    ap.add_argument("--out", type=str, default="rois.yaml", help="Archivo YAML de salida")
    ap.add_argument("--load", type=str, default=None, help="Cargar YAML existente para seguir editando")
    args = ap.parse_args()

    ts = None
    if args.timestamp:
        ts = parse_timecode(args.timestamp)

    frame = load_frame_from_source(args.source, frame_idx=args.frame, timestamp_s=ts)

    preload_polys, preload_ids = load_preload_yaml(args.load) if args.load else (None, None)

    editor = ROIEditor(frame, out_path=args.out, preload_polys=preload_polys, preload_ids=preload_ids)
    print("Controles: Left=add | Right/Enter=close | u=undo | d=del poly | s=save | q=quit")
    editor.run()

if __name__ == "__main__":
    main()