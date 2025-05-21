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
        self.tracking_failure_threshold = 10  # Frames before reinitializing tracker
        self.input_source = 0  # 0 for webcam, or file path for video
        self.reinitialize_interval = 30  # Frames between tracker reinitializations
        self.track_max_objects = 10  # Maximum number of objects to track simultaneously
        
# === Tracker Factory ===
def create_tracker_by_name(tracker_type):
    """Create a tracker instance based on name"""
    if hasattr(cv2, 'legacy'):
        legacy = cv2.legacy
    else:
        legacy = cv2
        
    if tracker_type == 'CSRT':
        return legacy.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        return legacy.TrackerKCF_create()
    elif tracker_type == 'MIL':
        return legacy.TrackerMIL_create()
    elif tracker_type == 'MOSSE':
        return legacy.TrackerMOSSE_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

def create_multi_tracker():
    """Create a multi-tracker instance compatible with the installed OpenCV version"""
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'MultiTracker_create'):
        return cv2.legacy.MultiTracker_create()
    elif hasattr(cv2, 'MultiTracker_create'):
        return cv2.MultiTracker_create()
    else:
        raise ImportError("MultiTracker not found. Install OpenCV-contrib with `pip install opencv-contrib-python`.")

class ObjectTracker:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config.model_path)
        os.makedirs(config.output_dir, exist_ok=True)
        self.object_colors = {}  # To assign consistent colors to objects
        self.frame_count = 0
        self.trajectories = defaultdict(list)
        self.tracking_failures = defaultdict(int)
        self.class_labels = {}  # To store class labels for tracked objects
        self.object_ids = {}    # To assign unique IDs to objects
        self.next_id = 1        # Counter for assigning unique IDs
        
    def get_object_color(self, obj_id):
        """Generate a consistent color for each object ID"""
        if obj_id not in self.object_colors:
            # Generate a random but visually distinct color
            self.object_colors[obj_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.object_colors[obj_id]
    
    def run_detection(self, frame):
        """Run YOLO detection on a frame and filter by confidence and class"""
        results = self.model(frame)[0]
        
        bboxes = []
        class_names = []
        confidences = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            
            # Filter by confidence and class if specified
            if conf >= self.config.confidence_threshold and (
                self.config.detection_classes is None or 
                cls_name in self.config.detection_classes
            ):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2 - x1, y2 - y1))  # Convert to (x, y, w, h)
                class_names.append(cls_name)
                confidences.append(conf)
        
        # Sort by confidence and take the top N objects
        if len(bboxes) > self.config.track_max_objects:
            indices = np.argsort(confidences)[::-1][:self.config.track_max_objects]
            bboxes = [bboxes[i] for i in indices]
            class_names = [class_names[i] for i in indices]
            confidences = [confidences[i] for i in indices]
            
        return bboxes, class_names, confidences
    
    def draw_tracking_info(self, frame, boxes, object_ids):
        """Draw bounding boxes, trajectories, and labels on frame"""
        for i, (box, obj_id) in enumerate(zip(boxes, object_ids)):
            if obj_id in self.tracking_failures and self.tracking_failures[obj_id] > self.config.tracking_failure_threshold:
                continue  # Skip drawing lost objects
                
            x, y, w, h = [int(v) for v in box]
            center = (x + w // 2, y + h // 2)
            color = self.get_object_color(obj_id)
            
            # Add current position to trajectory
            self.trajectories[obj_id].append(center)
            
            # Limit trajectory length to avoid clutter
            max_traj_length = 60  # ~2 seconds at 30fps
            if len(self.trajectories[obj_id]) > max_traj_length:
                self.trajectories[obj_id] = self.trajectories[obj_id][-max_traj_length:]
            
            # Draw box with thickness based on tracking quality
            thickness = 2
            if obj_id in self.tracking_failures:
                quality = 1.0 - (self.tracking_failures[obj_id] / self.config.tracking_failure_threshold)
                thickness = max(1, int(3 * quality))
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw trajectory line
            if len(self.trajectories[obj_id]) > 1:
                for j in range(1, len(self.trajectories[obj_id])):
                    cv2.line(frame, self.trajectories[obj_id][j-1], 
                             self.trajectories[obj_id][j], color, 2)
            
            # Add label with ID and class
            label = f"ID:{obj_id} - {self.class_labels.get(obj_id, 'Unknown')}"
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame information
        cv2.putText(frame, f"Frame: {self.frame_count} | Objects: {len(boxes)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def check_tracking_quality(self, boxes):
        """Check if tracking quality is acceptable"""
        tracking_quality = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            
            # Check if box is outside the frame or too small
            if x < 0 or y < 0 or w < 10 or h < 10:
                tracking_quality.append(False)
            else:
                tracking_quality.append(True)
                
        return tracking_quality
    
    def run(self):
        """Main tracking loop"""
        # Initialize video capture
        cap = cv2.VideoCapture(self.config.input_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.config.input_source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1:
            fps = 30  # Default if unable to determine
        
        # Initialize video writer if needed
        video_writer = None
        if self.config.output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                self.config.video_save_path,
                fourcc, 
                fps,
                (width, height)
            )
        
        # Initialize trackers
        multi_tracker = create_multi_tracker()
        tracked_object_ids = []
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
            
            self.frame_count += 1
            
            # Determine if we need to reinitialize trackers
            reinitialize = (self.frame_count % self.config.reinitialize_interval == 0) or not tracked_object_ids
            
            if reinitialize:
                print(f"Reinitializing trackers at frame {self.frame_count}")
                # Run object detection
                bboxes, class_names, confidences = self.run_detection(frame)
                
                if not bboxes:
                    print("No objects detected.")
                    # Continue showing frames but no tracking
                    cv2.putText(frame, "No objects detected", (width//2 - 100, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Reset tracker and create new one
                    multi_tracker = create_multi_tracker()
                    tracked_object_ids = []
                    
                    # Initialize trackers for each detected object
                    for i, (bbox, class_name) in enumerate(zip(bboxes, class_names)):
                        tracker = create_tracker_by_name(self.config.tracker_type)
                        
                        # Assign a new ID to this object
                        obj_id = self.next_id
                        self.next_id += 1
                        
                        # Store class label
                        self.class_labels[obj_id] = class_name
                        
                        # Add to multi-tracker
                        success = multi_tracker.add(tracker, frame, bbox)
                        if success:
                            tracked_object_ids.append(obj_id)
                            print(f"Started tracking {class_name} (ID: {obj_id}) with confidence {confidences[i]:.2f}")
                        else:
                            print(f"Failed to initialize tracker for {class_name}")
            
            # Update trackers
            success, boxes = multi_tracker.update(frame)
            
            # Check tracking quality
            if success:
                tracking_quality = self.check_tracking_quality(boxes)
                
                # Update tracking failures counter
                for i, (quality, obj_id) in enumerate(zip(tracking_quality, tracked_object_ids)):
                    if not quality:
                        self.tracking_failures[obj_id] += 1
                    else:
                        # Reset failure counter if tracking is good
                        self.tracking_failures[obj_id] = max(0, self.tracking_failures[obj_id] - 1)
            
            # Draw tracking information
            output_frame = self.draw_tracking_info(frame.copy(), boxes, tracked_object_ids)
            
            # Calculate and display FPS
            processing_time = time.time() - start_time
            fps_text = f"FPS: {1.0/processing_time:.1f}"
            cv2.putText(output_frame, fps_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow("Enhanced Object Tracker", output_frame)
            
            # Write frame to video if enabled
            if video_writer is not None:
                video_writer.write(output_frame)
            
            # Check for user exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User interrupted.")
                break
        
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"Tracking session completed. Output saved to {self.config.video_save_path if self.config.output_video else 'not saved'}")

def main():
    """Parse arguments and run the tracker"""
    parser = argparse.ArgumentParser(description="Enhanced Object Tracking System")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--tracker", type=str, default="CSRT", choices=["CSRT", "KCF", "MIL", "MOSSE"], 
                        help="Tracker algorithm to use")
    parser.add_argument("--input", type=str, default="0", help="Input source (0 for webcam, or video file path)")
    parser.add_argument("--output", type=str, default="./outputs/tracked_output.mp4", help="Output video path")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--classes", type=str, default=None, 
                        help="Comma-separated list of classes to track (e.g., 'person,car,dog')")
    parser.add_argument("--max-objects", type=int, default=10, help="Maximum number of objects to track")
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.model_path = args.model
    config.tracker_type = args.tracker
    config.video_save_path = args.output
    config.output_dir = os.path.dirname(args.output)
    config.confidence_threshold = args.confidence
    config.track_max_objects = args.max_objects
    
    # Handle input source
    try:
        config.input_source = int(args.input)  # Try as camera index
    except ValueError:
        config.input_source = args.input  # Use as file path
    
    # Handle classes
    if args.classes:
        config.detection_classes = args.classes.split(',')
        print(f"Tracking classes: {config.detection_classes}")
    
    # Create and run tracker
    tracker = ObjectTracker(config)
    tracker.run()

if __name__ == '__main__':
    main()
