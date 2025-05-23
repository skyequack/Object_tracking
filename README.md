Object Tracking

A robust real-time object tracking system built with OpenCV and YOLOv8 that can detect and track multiple objects simultaneously across video streams.

![Object Tracking Demo](./docs/demo.gif)

## Features

- **Multi-object tracking**: Track multiple objects simultaneously with unique IDs
- **Support for various object types**: Track any object detectable by YOLOv8 (80+ classes)
- **Trajectory visualization**: See the path of each tracked object
- **Tracking failure detection**: Automatic detection and recovery from lost tracks
- **Visual feedback**: Color-coded bounding boxes and object identification
- **Highly configurable**: Customize via command-line arguments
- **Performance metrics**: Real-time FPS calculation and display

## Requirements

- Python 3.7+
- OpenCV (with contrib modules)
- PyTorch
- Ultralytics YOLOv8

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/skyequack/object-tracking.git
   cd object-tracking
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python3 multi_object_tracker.py
```

This will start the tracker using your webcam and the default configuration.

### Advanced Options

```bash
python3 multi_object_tracker.py --input 0 --model yolov8n.pt --tracker CSRT --classes person,car --confidence 0.5
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to YOLO model | `yolov8n.pt` |
| `--tracker` | Tracker algorithm (CSRT, KCF, MIL, MOSSE) | `CSRT` |
| `--input` | Input source (0 for webcam, or video file path) | `0` |
| `--output` | Output video path | `./outputs/tracked_output.mp4` |
| `--confidence` | Detection confidence threshold | `0.5` |
| `--classes` | Comma-separated list of classes to track | None (all classes) |
| `--max-objects` | Maximum number of objects to track | `10` |

## Project Structure

```
enhanced-object-tracker/
├── enhanced_object_tracker.py    # Main script
├── outputs/                      # Directory for output videos
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## How It Works

1. **Object Detection**: Uses YOLOv8 to detect objects in frames
2. **Object Tracking**: Uses OpenCV tracking algorithms to follow objects between detections
3. **Trajectory Tracking**: Records and displays the path of each tracked object
4. **Failure Detection**: Monitors tracking quality and reinitializes when needed
5. **Visualization**: Presents tracking information with bounding boxes and labels

## Tracking Algorithms

The system supports multiple tracking algorithms:

- **CSRT**: More accurate but slower (default)
- **KCF**: Balanced performance
- **MIL**: Better recovery capabilities but slower
- **MOSSE**: Fastest but less accurate

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
