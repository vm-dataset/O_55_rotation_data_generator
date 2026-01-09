"""
Prompts for 3D Mental Rotation Tasks

This module provides prompt generation for 3D mental rotation tasks.
Prompts describe camera rotation around fixed 3D voxel structures.
"""

# Standardized prompt template for rotation tasks
PROMPT_TEMPLATE = (
    "A {num_voxels}-block sculpture sits fixed on a table. "
    "First frame: Your camera is tilted at {elev1}° elevation, viewing from {azim1}° azimuth. "
    "Final frame: Your camera remains at {elev2}° elevation, but rotates horizontally to {azim2}° azimuth. This is a 180-degree rotation "
    "Create a smooth video showing the camera's horizontal rotation around the sculpture, and try to maintain the tilted viewing angle throughout."
)


def get_prompt(
    num_voxels: int,
    elev1: float,
    azim1: float,
    elev2: float,
    azim2: float
) -> str:
    """
    Generate prompt for a 3D mental rotation task.
    
    Args:
        num_voxels: Number of voxels in the structure (8 or 9)
        elev1: Initial elevation angle in degrees
        azim1: Initial azimuth angle in degrees
        elev2: Final elevation angle in degrees (should match elev1 for horizontal rotation)
        azim2: Final azimuth angle in degrees (should be elev1 + 180°)
        
    Returns:
        Formatted prompt string
    """
    return PROMPT_TEMPLATE.format(
        num_voxels=num_voxels,
        elev1=int(elev1),
        azim1=int(azim1),
        elev2=int(elev2),
        azim2=int(azim2)
    )
