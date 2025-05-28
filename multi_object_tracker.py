import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import argparse
from collections import defaultdict

# === Settings ===
class Config:
    def __init__(self):
        self.model_path = 'yolov8n.pt'  # Can be modified to use yolov8s.pt, yolov8m.pt for better accuracy
        self.tracker_type = 'CSRT'  # Options: 'CSRT', 'KCF', 'MIL', 'MOSSE'
        self.output_video = True
        self.output_dir = './outputs'
        self.video_save_path = os.path.join(self.output_dir, 'tracked_output.mp4')
        self.detection_classes = None  # None means track all detected objects
        self.confidence_threshold = 0.5
        self.tracking_failure_threshold = 10  # Frames before considering tracking lost
        self.input_source = 0  # 0 for webcam, or file path for video
        self.track_max_objects = 5  # Maximum number of objects to track simultaneously
        self.trajectory_length = 50  # Maximum number of points in trajectory
        self.interactive_mode = True  # Enable interactive object selection
        
# === Tracker Factory ===
def create_tracker_by_name(tracker_type):
    """Create a tracker instance based on name"""
    try:
        # Try legacy trackers first (newer OpenCV versions)
        if hasattr(cv2, 'legacy'):
            if tracker_type == 'CSRT':
                return cv2.legacy.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                return cv2.legacy.TrackerKCF_create()
            elif tracker_type == 'MIL':
                return cv2.legacy.TrackerMIL_create()
            elif tracker_type == 'MOSSE':
                return cv2.legacy.TrackerMOSSE_create()
        
        # Try direct cv2 trackers (older OpenCV versions)
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        elif tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
            
    except AttributeError:
        print(f"Warning: {tracker_type} tracker not available in this OpenCV version")
        print("Available trackers might be limited. Please install opencv-contrib-python")
        
        # Fallback to any available tracker
        try:
            if hasattr(cv2, 'legacy'):
                return cv2.legacy.TrackerCSRT_create()
            else:
                return cv2.TrackerCSRT_create()
        except:
            pass
    
    raise ValueError(f"No suitable tracker found. Please install opencv-contrib-python")

class SingleObjectTracker:
    """Tracks a single object with its own tracker instance"""
    def __init__(self, tracker_type, bbox, frame, obj_id, class_name, confidence):
        self.tracker = create_tracker_by_name(tracker_type)
        self.obj_id = obj_id
        self.class_name = class_name
        self.confidence = confidence
        self.trajectory = []
        self.tracking_failures = 0
        self.is_lost = False
        self.color = tuple(map(int, np.random.randint(50, 255, 3)))
        
        # Initialize tracker
        success = self.tracker.init(frame, bbox)
        if not success:
            raise Exception(f"Failed to initialize tracker for object {obj_id}")
        
        # Add initial position to trajectory
        x, y, w, h = bbox
        center = (int(x + w/2), int(y + h/2))
        self.trajectory.append(center)
        
    def update(self, frame, failure_threshold):
        """Update tracker and return success status and bounding box"""
        success, bbox = self.tracker.update(frame)
        
        if success:
            # Validate bounding box
            x, y, w, h = bbox
            if x < 0 or y < 0 or w < 10 or h < 10 or x > frame.shape[1] or y > frame.shape[0]:
                success = False
                
        if success:
            self.tracking_failures = 0
            self.is_lost = False
            
            # Update trajectory
            x, y, w, h = bbox
            center = (int(x + w/2), int(y + h/2))
            self.trajectory.append(center)
            
            # Limit trajectory length
            if len(self.trajectory) > 50:
                self.trajectory.pop(0)
                
            return True, bbox
        else:
            self.tracking_failures += 1
            if self.tracking_failures >= failure_threshold:
                self.is_lost = True
            return False, None
    
    def draw(self, frame):
        """Draw bounding box and trajectory on frame"""
        if self.is_lost:
            return
            
        # Draw trajectory
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                thickness = max(1, int(3 * (i / len(self.trajectory))))
                cv2.line(frame, self.trajectory[i-1], self.trajectory[i], self.color, thickness)
        
        # Draw current position if we have a recent position
        if self.trajectory:
            current_pos = self.trajectory[-1]
            cv2.circle(frame, current_pos, 5, self.color, -1)

class MultiObjectTracker:
    def __init__(self, config):
        self.config = config
        
        # Check if YOLO model file exists
        if not os.path.exists(config.model_path):
            print(f"Warning: Model file {config.model_path} not found.")
            print("YOLO will attempt to download the model automatically.")
        
        try:
            self.model = YOLO(config.model_path)
            print(f"✓ Loaded YOLO model: {config.model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Please ensure ultralytics is installed: pip install ultralytics")
            raise
            
        os.makedirs(config.output_dir, exist_ok=True)
        
        self.trackers = {}  # Dictionary of active trackers
        self.next_id = 1
        self.frame_count = 0
        self.mouse_callback_active = False
        self.selected_objects = []  # For interactive selection
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for object selection"""
        if event == cv2.EVENT_LBUTTONDOWN and self.mouse_callback_active:
            frame, detections = param
            
            # Find clicked detection
            for i, (bbox, class_name, confidence) in enumerate(detections):
                bx, by, bw, bh = bbox
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.selected_objects.append((bbox, class_name, confidence))
                    print(f"Selected {class_name} for tracking (confidence: {confidence:.2f})")
                    break
    
    def detect_objects(self, frame):
        """Run YOLO detection and return filtered results"""
        results = self.model(frame)[0]
        
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            
            if conf >= self.config.confidence_threshold and (
                self.config.detection_classes is None or 
                cls_name in self.config.detection_classes
            ):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                detections.append((bbox, cls_name, conf))
        
        # Sort by confidence and limit number
        detections.sort(key=lambda x: x[2], reverse=True)
        return detections[:self.config.track_max_objects]
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes for selection"""
        for bbox, class_name, confidence in detections:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Click on objects to track, press SPACE to start tracking", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Selected: {len(self.selected_objects)} objects", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def initialize_trackers(self, frame):
        """Initialize trackers for selected objects"""
        print("\n=== Initializing Trackers ===")
        for bbox, class_name, confidence in self.selected_objects:
            try:
                tracker = SingleObjectTracker(
                    self.config.tracker_type, bbox, frame, 
                    self.next_id, class_name, confidence
                )
                self.trackers[self.next_id] = tracker
                print(f"✓ Started tracking {class_name} (ID: {self.next_id})")
                self.next_id += 1
            except Exception as e:
                print(f"✗ Failed to initialize tracker for {class_name}: {e}")
        
        self.selected_objects.clear()
        print(f"Total active trackers: {len(self.trackers)}")
    
    def update_trackers(self, frame):
        """Update all active trackers"""
        lost_trackers = []
        
        for obj_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame, self.config.tracking_failure_threshold)
            
            if tracker.is_lost:
                lost_trackers.append(obj_id)
                print(f"⚠️  TRACKING LOST: {tracker.class_name} (ID: {obj_id})")
        
        # Remove lost trackers
        for obj_id in lost_trackers:
            del self.trackers[obj_id]
        
        return len(lost_trackers) > 0
    
    def draw_tracking_info(self, frame):
        """Draw all tracking information on frame"""
        # Draw individual tracker info
        for tracker in self.trackers.values():
            tracker.draw(frame)
        
        # Draw status information
        active_count = len(self.trackers)
        cv2.putText(frame, f"Active Trackers: {active_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'r' to redetect, 'q' to quit", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Alert if no active trackers
        if active_count == 0:
            cv2.putText(frame, "NO ACTIVE TRACKERS - Press 'r' to redetect", 
                       (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def run(self):
        """Main tracking loop"""
        print("=== Enhanced Multi-Object Tracker ===")
        print("Instructions:")
        print("- 'r': Re-detect and select objects")
        print("- 'q': Quit")
        print("- During selection: Click objects, then press SPACE to start tracking")
        
        # Initialize video capture
        cap = cv2.VideoCapture(self.config.input_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.config.input_source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer
        video_writer = None
        if self.config.output_video:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(self.config.video_save_path, fourcc, fps, (width, height))
                print(f"✓ Video output enabled: {self.config.video_save_path}")
            except Exception as e:
                print(f"Warning: Could not initialize video writer: {e}")
                self.config.output_video = False
        
        # Set up window
        window_name = "Enhanced Multi-Object Tracker"
        cv2.namedWindow(window_name)
        
        # Initial detection phase
        selection_mode = True
        
        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream.")
                    break
                
                self.frame_count += 1
                
                if selection_mode:
                    # Detection and selection phase
                    try:
                        detections = self.detect_objects(frame)
                        self.draw_detections(frame, detections)
                        
                        # Set up mouse callback for selection
                        self.mouse_callback_active = True
                        cv2.setMouseCallback(window_name, self.mouse_callback, (frame, detections))
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(' ') and self.selected_objects:
                            # Start tracking selected objects
                            self.initialize_trackers(frame)
                            selection_mode = False
                            self.mouse_callback_active = False
                        elif key == ord('q'):
                            break
                    except Exception as e:
                        print(f"Error during detection: {e}")
                        cv2.putText(frame, f"Detection Error: {str(e)[:50]}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # Tracking phase
                    try:
                        tracking_lost = self.update_trackers(frame)
                        self.draw_tracking_info(frame)
                        
                        # Handle user input
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('r'):
                            # Re-enter selection mode
                            selection_mode = True
                            self.trackers.clear()
                            print("\n=== Re-detection Mode ===")
                        elif key == ord('q'):
                            break
                    except Exception as e:
                        print(f"Error during tracking: {e}")
                        cv2.putText(frame, f"Tracking Error: {str(e)[:50]}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Calculate and display FPS
                processing_time = time.time() - start_time
                if processing_time > 0:
                    fps_text = f"FPS: {1.0/processing_time:.1f}"
                    cv2.putText(frame, fps_text, (width - 120, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Save frame if video output is enabled
                if video_writer is not None:
                    video_writer.write(frame)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            # Cleanup
            cap.release()
            if video_writer is not None:
                video_writer.release()
                print(f"Video saved to: {self.config.video_save_path}")
            cv2.destroyAllWindows()
            
            print("Tracking session completed.")

def main():
    """Parse arguments and run the tracker"""
    parser = argparse.ArgumentParser(description="Enhanced Multi-Object Tracking System")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--tracker", type=str, default="CSRT", 
                        choices=["CSRT", "KCF", "MIL", "MOSSE"], 
                        help="Tracker algorithm to use")
    parser.add_argument("--input", type=str, default="0", 
                        help="Input source (0 for webcam, or video file path)")
    parser.add_argument("--output", type=str, default="./outputs/tracked_output.mp4", 
                        help="Output video path")
    parser.add_argument("--confidence", type=float, default=0.5, 
                        help="Detection confidence threshold")
    parser.add_argument("--classes", type=str, default=None, 
                        help="Comma-separated list of classes to track (e.g., 'person,car,dog')")
    parser.add_argument("--max-objects", type=int, default=5, 
                        help="Maximum number of objects to track")
    parser.add_argument("--no-output", action="store_true", 
                        help="Disable video output saving")
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.model_path = args.model
    config.tracker_type = args.tracker
    config.video_save_path = args.output
    config.output_dir = os.path.dirname(args.output)
    config.confidence_threshold = args.confidence
    config.track_max_objects = args.max_objects
    config.output_video = not args.no_output
    
    # Handle input source
    try:
        config.input_source = int(args.input)
    except ValueError:
        config.input_source = args.input
    
    # Handle classes
    if args.classes:
        config.detection_classes = args.classes.split(',')
        print(f"Tracking classes: {config.detection_classes}")
    
    # Create and run tracker
    tracker = MultiObjectTracker(config)
    tracker.run()

if __name__ == '__main__':
    main()
