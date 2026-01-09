# Rotation Task Data Generator 🔄

A data generator for creating 3D mental rotation tasks. Generates 3D voxel structures (snake-like configurations) and creates rotation tasks where models must demonstrate spatial reasoning by showing how these structures appear when the camera rotates horizontally around them.

Repository: [O_55_rotation_data_generator](https://github.com/vm-dataset/O_55_rotation_data_generator)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vm-dataset/O_55_rotation_data_generator.git
cd O_55_rotation_data_generator

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Generate tasks
python examples/generate.py --num-samples 50
```

---

## 📁 Structure

```
rotation-task-data-generator/
├── core/                    # Framework utilities
│   ├── base_generator.py   # Abstract base class
│   ├── schemas.py          # Pydantic models
│   ├── image_utils.py      # Image helpers
│   ├── video_utils.py      # Video generation
│   └── output_writer.py    # File output
├── src/                     # Rotation task implementation
│   ├── generator.py        # 3D voxel rotation generator
│   ├── prompts.py          # Prompt templates
│   └── config.py           # Configuration
├── examples/
│   └── generate.py         # Entry point
└── data/questions/         # Generated output
```

---

## 📦 Output Format

Every generator produces:

```
data/questions/rotation_task/{task_id}/
├── first_frame.png          # Initial viewpoint (REQUIRED)
├── final_frame.png          # Rotated viewpoint (REQUIRED)
├── prompt.txt               # Instructions (REQUIRED)
└── ground_truth.mp4         # Rotation video (OPTIONAL)
```

### Output Details

- **first_frame.png**: 3D voxel structure viewed from initial angle (20-40° elevation, random azimuth)
- **final_frame.png**: Same structure viewed from rotated angle (same elevation, azimuth + 180°)
- **prompt.txt**: Natural language instructions describing the camera rotation
- **ground_truth.mp4**: Smooth video showing the camera rotation (if enabled)

---

## 🎯 Task Description

### Core Challenge

The 3D Mental Rotation Task evaluates spatial reasoning by requiring models to:

1. **Parse 3D Structure**: Understand the 3D configuration from a tilted 2D projection
2. **Horizontal Rotation**: Generate smooth camera rotation around the fixed object
3. **Perspective Consistency**: Maintain consistent tilted viewing angle throughout
4. **Generate Transition**: Create smooth video showing 180° horizontal camera movement

### Visual Elements

- **3D Voxel Structures**: Snake-like configurations made of 8-9 connected cubes
- **Tilted Views**: Consistent 20-40° elevation for clear 3D perspective
- **Horizontal Rotations**: Camera moves horizontally around the fixed sculpture (180° change)
- **Consistent Rendering**: High-quality 3D visualization with proper lighting

---

## ⚙️ Configuration

All configuration is in `src/config.py`. Key parameters:

```python
class TaskConfig(GenerationConfig):
    # Domain
    domain: str = "rotation"
    image_size: tuple[int, int] = (400, 400)
    
    # Video settings
    generate_videos: bool = True
    video_fps: int = 10
    
    # Voxel structure parameters
    num_voxels_range: tuple[int, int] = (8, 9)
    
    # Viewpoint parameters
    elevation_range: tuple[int, int] = (20, 40)  # degrees
    rotation_angle: int = 180  # degrees (fixed)
    
    # Snake generation parameters
    snake_lmin: int = 1
    snake_lmax: int = 3
    snake_p_branch: float = 0.2
    snake_max_deg: int = 3
    snake_tries: int = 1000
```

---

## 📝 Usage Examples

### Basic Generation

```bash
# Generate 50 rotation tasks
python examples/generate.py --num-samples 50

# Generate with custom output directory
python examples/generate.py --num-samples 100 --output data/my_rotation_tasks

# Generate without videos
python examples/generate.py --num-samples 50 --no-videos

# Generate with fixed random seed
python examples/generate.py --num-samples 50 --seed 42
```

### Programmatic Usage

```python
from pathlib import Path
from src import TaskGenerator, TaskConfig
from core import OutputWriter

# Configure
config = TaskConfig(
    num_samples=50,
    random_seed=42,
    output_dir=Path("data/questions"),
    generate_videos=True
)

# Generate
generator = TaskGenerator(config)
tasks = generator.generate_dataset()

# Write to disk
writer = OutputWriter(Path("data/questions"))
writer.write_dataset(tasks)
```

---

## 🔧 Dependencies

Core dependencies:
- `numpy>=1.21.0` - Array operations
- `Pillow==10.4.0` - Image processing
- `pydantic==2.10.5` - Data validation
- `matplotlib>=3.5.0` - 3D rendering
- `opencv-python==4.10.0.84` - Video generation (optional)

---

## 📊 Generated Data Characteristics

- **Voxel Count**: 8-9 cubes per structure
- **Structure Type**: Snake-like connected 3D configurations
- **View Angles**: 
  - Elevation: 20-40° (tilted for 3D perspective)
  - Azimuth: 0-359° initial, +180° for final view
- **Image Format**: 400×400 RGB PNG
- **Video Format**: MP4 (H.264) at 10 fps (if enabled)

---

## 🎨 Customization

To customize the rotation task:

1. **Adjust Voxel Count**: Modify `num_voxels_range` in `src/config.py`
2. **Change View Angles**: Modify `elevation_range` and `rotation_angle` in `src/config.py`
3. **Modify Prompts**: Edit `src/prompts.py` to change instruction templates
4. **Adjust Structure Complexity**: Modify snake generation parameters in `src/config.py`

---

## 📚 Technical Details

### Voxel Generation Algorithm

The generator uses a sophisticated recursive algorithm to create 3D snake-like structures:

- Starts with a single voxel at origin
- Grows straight segments of length Lmin to Lmax
- Optionally branches perpendicular to current direction
- Ensures all three axes (x, y, z) are utilized
- Validates structure is not rotationally symmetric

### Rendering Pipeline

1. **3D Rendering**: Uses matplotlib 3D with perspective projection
2. **Cube Styling**: Light blue cubes (RGB: 0.7, 0.7, 0.9) with black edges
3. **Image Processing**: Crops to square, resizes to target size, converts to RGB
4. **Video Generation**: Creates smooth interpolation between viewpoints

---

## 📄 License

[Add your license here]

---

## 🙏 Acknowledgments

This project is based on the VMEvalKit rotation task implementation.
