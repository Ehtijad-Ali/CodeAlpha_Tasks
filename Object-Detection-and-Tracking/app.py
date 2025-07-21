import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import time

# Try to import and set a modern theme
try:
    from ttkthemes import ThemedTk
except ImportError:
    print("ttkthemes not found. Using standard Tkinter theme.")
    print("For a better look, run: pip install ttkthemes")
    ThemedTk = tk.Tk

# ==============================================================================
#  Object Detection and Tracking Classes (Modified to accept drawing options)
# ==============================================================================

class SimpleTracker:
    # --- This class is unchanged from your original code ---
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    def deregister(self, object_id):
        if object_id in self.objects: del self.objects[object_id]
        if object_id in self.disappeared: del self.disappeared[object_id]
    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return list(self.objects.items())
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            if len(object_centroids) > 0:
                D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
                rows, cols = D.min(axis=1).argsort(), D.argmin(axis=1)[D.min(axis=1).argsort()]
                used_rows, used_cols = set(), set()
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols or D[row, col] > 50:
                        continue
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)
                unused_rows = set(range(0, D.shape[0])).difference(used_rows)
                unused_cols = set(range(0, D.shape[1])).difference(used_cols)
                if D.shape[0] >= D.shape[1]:
                    for row in unused_rows:
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister(object_id)
                else:
                    for col in unused_cols:
                        self.register(input_centroids[col])
        return list(self.objects.items())

class ObjectDetectionTracker:
    def __init__(self):
        self.model = None
        self.conf_threshold = 0.5
        self.tracker = SimpleTracker()
        self.track_history = defaultdict(lambda: [])
        self.class_names = {}
    
    def set_model(self, model_path):
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print("Model loaded successfully.")

    def set_confidence(self, conf_threshold):
        self.conf_threshold = conf_threshold

    def process_frame(self, frame, draw_boxes=True, draw_trails=True):
        if self.model is None:
            return frame

        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append({'bbox': [x1, y1, x2, y2], 'class_name': self.class_names.get(cls, f'cls_{cls}')})
        
        rects = [d['bbox'] for d in detections]
        tracked_objects = self.tracker.update(rects)
        annotated_frame = frame.copy()
        
        for object_id, centroid in tracked_objects:
            # Associate tracked object with a detection box for drawing
            best_match = None
            min_dist = float('inf')
            for det in detections:
                cx = int((det['bbox'][0] + det['bbox'][2]) / 2)
                cy = int((det['bbox'][1] + det['bbox'][3]) / 2)
                dist = np.linalg.norm((cx - centroid[0], cy - centroid[1]))
                if dist < min_dist:
                    min_dist = dist
                    best_match = det
            
            if best_match and min_dist < 50:
                x1, y1, x2, y2 = best_match['bbox']
                class_name = best_match['class_name']
                label = f"ID:{object_id} {class_name}"
                
                if draw_boxes:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if draw_trails:
                self.track_history[object_id].append(centroid)
                if len(self.track_history[object_id]) > 30:
                    self.track_history[object_id].pop(0)
                points = np.array(self.track_history[object_id], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)
        return annotated_frame


# ==============================================================================
#  ADVANCED TKINTER GUI APPLICATION
# ==============================================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced YOLO Object Tracker")
        self.root.geometry("1200x800")
        self.detector = ObjectDetectionTracker()

        # --- State Variables ---
        self.video_source = None
        self.cap = None
        self.running = False
        self.paused = False
        self.total_frames = 0
        self.current_frame = 0
        self.fps_start_time = 0
        self.fps_frame_count = 0
        self.fps = 0

        # --- UI Control Variables ---
        self.model_var = tk.StringVar(value='yolov8n.pt')
        self.conf_var = tk.DoubleVar(value=0.5)
        self.draw_boxes_var = tk.BooleanVar(value=True)
        self.draw_trails_var = tk.BooleanVar(value=True)
        
        # --- UI Creation ---
        self._create_widgets()

    def _create_widgets(self):
        # --- Main Layout (Paned Window) ---
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Left Panel (Video Display) ---
        video_frame = ttk.Frame(paned_window, width=900, height=750)
        self.video_label = ttk.Label(video_frame, anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        paned_window.add(video_frame, weight=4)

        # --- Right Panel (Controls) ---
        control_panel = ttk.Frame(paned_window, width=300)
        control_panel.pack_propagate(False)
        paned_window.add(control_panel, weight=1)

        # --- Control Widgets inside Right Panel ---
        # Model Selection
        lf_model = ttk.LabelFrame(control_panel, text="Model & Source")
        lf_model.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(lf_model, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_combo = ttk.Combobox(lf_model, textvariable=self.model_var, values=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'])
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        self.btn_webcam = ttk.Button(lf_model, text="▶ Start Webcam", command=self._start_webcam)
        self.btn_webcam.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        
        self.btn_video = ttk.Button(lf_model, text="📂 Open Video File", command=self._open_video_file)
        self.btn_video.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        # Detection Settings
        lf_settings = ttk.LabelFrame(control_panel, text="Detection Settings")
        lf_settings.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(lf_settings, text="Confidence:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.conf_scale = ttk.Scale(lf_settings, from_=0.1, to=0.9, orient=tk.HORIZONTAL, variable=self.conf_var, command=lambda s: self.detector.set_confidence(self.conf_var.get()))
        self.conf_scale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        self.cb_boxes = ttk.Checkbutton(lf_settings, text="Draw Bounding Boxes", variable=self.draw_boxes_var)
        self.cb_boxes.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        self.cb_trails = ttk.Checkbutton(lf_settings, text="Draw Tracking Trails", variable=self.draw_trails_var)
        self.cb_trails.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        # Live Stats
        lf_stats = ttk.LabelFrame(control_panel, text="Live Statistics")
        lf_stats.pack(fill=tk.X, padx=10, pady=10)
        
        self.label_fps = ttk.Label(lf_stats, text="FPS: -")
        self.label_fps.pack(padx=5, pady=2, anchor=tk.W)
        
        self.label_objects = ttk.Label(lf_stats, text="Tracked Objects: -")
        self.label_objects.pack(padx=5, pady=2, anchor=tk.W)

        # --- Bottom Controls and Status Bar ---
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_pause = ttk.Button(bottom_frame, text="Pause", command=self._pause_resume, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(bottom_frame, text="Stop", command=self._stop_processing, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.status_bar = ttk.Label(self.root, text="Status: Idle", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _start_processing(self):
        try:
            self.detector.set_model(self.model_var.get())
            self.detector.set_confidence(self.conf_var.get())
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened(): raise ValueError("Unable to open video source")
        except Exception as e:
            self.status_bar.config(text=f"Error: {e}")
            return
            
        self.running = True
        self.paused = False
        self.btn_webcam.config(state=tk.DISABLED)
        self.btn_video.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.NORMAL, text="Pause")
        
        if isinstance(self.video_source, str): # Video file
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar.config(maximum=self.total_frames)
        else: # Webcam
            self.total_frames = float('inf')
            self.progress_bar.config(maximum=100, mode='indeterminate')
            self.progress_bar.start()

        self.current_frame = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self._video_loop()

    def _video_loop(self):
        if not self.running: return

        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
                
                # Process frame
                annotated_frame = self.detector.process_frame(frame, self.draw_boxes_var.get(), self.draw_trails_var.get())
                
                # Update UI
                self._update_ui(annotated_frame)
            else:
                self._stop_processing()
                self.status_bar.config(text="Status: Video finished")
                return
        
        self.root.after(10, self._video_loop)

    def _update_ui(self, frame):
        # Update FPS
        self.fps_frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time > 1:
            self.fps = self.fps_frame_count / elapsed_time
            self.label_fps.config(text=f"FPS: {self.fps:.2f}")
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

        # Update object count
        self.label_objects.config(text=f"Tracked Objects: {len(self.detector.tracker.objects)}")
        
        # Update progress bar
        if self.total_frames != float('inf'):
            self.progress_bar['value'] = self.current_frame
        
        # Update video label
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

    def _start_webcam(self):
        self.video_source = 0
        self.status_bar.config(text="Status: Starting Webcam...")
        self._start_processing()

    def _open_video_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if filepath:
            self.video_source = filepath
            self.status_bar.config(text=f"Status: Opening {filepath.split('/')[-1]}...")
            self._start_processing()

    def _stop_processing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.btn_webcam.config(state=tk.NORMAL)
        self.btn_video.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.DISABLED, text="Pause")
        self.progress_bar.stop()
        self.progress_bar['value'] = 0
        self.status_bar.config(text="Status: Stopped")
        self.video_label.config(image='')
        self.label_fps.config(text="FPS: -")
        self.label_objects.config(text="Tracked Objects: -")

    def _pause_resume(self):
        self.paused = not self.paused
        self.btn_pause.config(text="Resume" if self.paused else "Pause")
        self.status_bar.config(text="Status: Paused" if self.paused else "Status: Running...")

    def _on_closing(self):
        self._stop_processing()
        self.root.destroy()

if __name__ == "__main__":
    try:
        # Use a themed Tk window if available
        root = ThemedTk(theme="equilux") # Other good themes: "arc", "plastik"
    except Exception:
        root = tk.Tk()
        
    app = App(root)
    root.mainloop()