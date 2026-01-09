"""
3D Mental Rotation Task Configuration

Configuration for generating 3D mental rotation tasks with voxel structures.
"""

from pydantic import Field
from core import GenerationConfig


class TaskConfig(GenerationConfig):
    """
    3D Mental Rotation Task configuration.
    
    Inherited from GenerationConfig:
        - num_samples: int          # Number of samples to generate
        - domain: str               # Task domain name
        - difficulty: Optional[str] # Difficulty level
        - random_seed: Optional[int] # For reproducibility
        - output_dir: Path          # Where to save outputs
        - image_size: tuple[int, int] # Image dimensions
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    #  OVERRIDE DEFAULTS
    # ══════════════════════════════════════════════════════════════════════════
    
    domain: str = Field(default="rotation")
    image_size: tuple[int, int] = Field(default=(400, 400))
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    generate_videos: bool = Field(
        default=True,
        description="Whether to generate ground truth videos"
    )
    
    video_fps: int = Field(
        default=10,
        description="Video frame rate"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  ROTATION TASK-SPECIFIC SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    # Voxel structure parameters
    num_voxels_range: tuple[int, int] = Field(
        default=(8, 9),
        description="Range of voxel counts (min, max)"
    )
    
    # Viewpoint parameters
    elevation_range: tuple[int, int] = Field(
        default=(20, 40),
        description="Elevation angle range in degrees (tilted view)"
    )
    
    rotation_angle: int = Field(
        default=180,
        description="Fixed horizontal rotation angle in degrees"
    )
    
    # Snake generation parameters
    snake_lmin: int = Field(
        default=1,
        description="Minimum segment length for voxel snake"
    )
    
    snake_lmax: int = Field(
        default=3,
        description="Maximum segment length for voxel snake"
    )
    
    snake_p_branch: float = Field(
        default=0.2,
        description="Branching probability for voxel snake"
    )
    
    snake_max_deg: int = Field(
        default=3,
        description="Maximum neighbors per voxel"
    )
    
    snake_tries: int = Field(
        default=1000,
        description="Maximum attempts for generating a valid snake structure"
    )
