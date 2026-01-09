"""
3D Mental Rotation Task Generator

This module generates 3D voxel structures (snake-like configurations) and creates
mental rotation tasks where models must demonstrate spatial reasoning by showing
how these structures appear when the camera rotates horizontally around them.
"""

import math
import random
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Iterable, Optional
from itertools import permutations, product

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from core import BaseGenerator, TaskPair, ImageRenderer
from core.video_utils import VideoGenerator
from .config import TaskConfig
from .prompts import get_prompt

# Constants
DIRS: List[Tuple[int, int, int]] = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

Voxel = Tuple[int, int, int]


class TaskGenerator(BaseGenerator):
    """
    3D Mental Rotation Task Generator.
    
    Generates 3D voxel structures and creates rotation tasks with tilted camera views
    and 180° horizontal rotations.
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.renderer = ImageRenderer(image_size=config.image_size)
        
        # Initialize video generator if enabled
        self.video_generator = None
        if config.generate_videos and VideoGenerator.is_available():
            self.video_generator = VideoGenerator(fps=config.video_fps, output_format="mp4")
        
        # Figure size for matplotlib rendering
        self.figure_size = (8, 8)
    
    def generate_task_pair(self, task_id: str) -> TaskPair:
        """Generate one rotation task pair."""
        
        # Generate voxel structure
        num_voxels = random.randint(*self.config.num_voxels_range)
        voxels = self._generate_snake(
            N=num_voxels,
            Lmin=self.config.snake_lmin,
            Lmax=self.config.snake_lmax,
            p_branch=self.config.snake_p_branch,
            max_deg=self.config.snake_max_deg,
            tries=self.config.snake_tries
        )
        
        # Generate viewpoints (horizontal rotation with tilted view)
        tilted_elevation = random.randint(*self.config.elevation_range)
        azim1 = random.randint(0, 359)
        azim2 = (azim1 + self.config.rotation_angle) % 360
        
        elev1, elev2 = tilted_elevation, tilted_elevation
        
        # Render images
        first_image = self._render_voxel_image(voxels, elev1, azim1)
        final_image = self._render_voxel_image(voxels, elev2, azim2)
        
        # Generate video (optional)
        video_path = None
        if self.config.generate_videos and self.video_generator:
            video_path = self._generate_video(first_image, final_image, task_id, voxels, elev1, azim1, elev2, azim2)
        
        # Generate prompt
        prompt = get_prompt(
            num_voxels=len(voxels),
            elev1=elev1,
            azim1=azim1,
            elev2=elev2,
            azim2=azim2
        )
        
        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=prompt,
            first_image=first_image,
            final_image=final_image,
            ground_truth_video=video_path
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VOXEL GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_snake(
        self,
        N: int,
        Lmin: int,
        Lmax: int,
        p_branch: float,
        max_deg: int,
        tries: int,
        rng: Optional[random.Random] = None,
    ) -> List[Voxel]:
        """Create a 3-D voxel snake."""
        if rng is None:
            rng = random.Random()

        for _ in range(tries):
            voxels: Set[Voxel] = {(0, 0, 0)}
            order: List[Voxel] = [(0, 0, 0)]
            axes_used: Set[str] = set()

            # Choose initial heading
            d = rng.choice(DIRS)
            axes_used.add(self._axis_of(d))

            while len(voxels) < N:
                # Grow main straight segment
                seg_len = min(rng.randint(Lmin, Lmax), N - len(voxels))
                x, y, z = order[-1]
                main_path: List[Voxel] = []

                for _ in range(seg_len):
                    x += d[0]; y += d[1]; z += d[2]
                    nxt = (x, y, z)

                    if nxt in voxels:
                        break
                    if self._neighbour_count(nxt, voxels) >= max_deg:
                        break
                    if any(
                        self._neighbour_count(nbr, voxels) + 1 > max_deg
                        for nbr in ((x + dx, y + dy, z + dz) for dx, dy, dz in DIRS)
                        if nbr in voxels
                    ):
                        break
                    main_path.append(nxt)
                else:
                    voxels.update(main_path)
                    order.extend(main_path)
                    axes_used.add(self._axis_of(d))

                if len(main_path) < seg_len:
                    break

                if len(voxels) >= N:
                    break

                # Optional branch from segment start
                if rng.random() < p_branch and len(voxels) < N and len(main_path) > 0:
                    seg_start_idx = len(order) - len(main_path) - 1
                    if seg_start_idx >= 0:
                        sx, sy, sz = order[seg_start_idx]
                        possible_branches = self._orthogonals(d)
                        if possible_branches:
                            branch_dir = rng.choice(possible_branches)
                            bx, by, bz = sx + branch_dir[0], sy + branch_dir[1], sz + branch_dir[2]
                            br_vox = (bx, by, bz)
                            if (
                                br_vox not in voxels
                                and self._neighbour_count(br_vox, voxels) < max_deg
                                and self._neighbour_count((sx, sy, sz), voxels) + 1 <= max_deg
                            ):
                                voxels.add(br_vox)
                                order.append(br_vox)
                                axes_used.add(self._axis_of(branch_dir))

                # Choose next heading
                orths = self._orthogonals(d)
                rng.shuffle(orths)

                unused_orths = [v for v in orths if self._axis_of(v) not in axes_used]
                for nd in unused_orths + orths:
                    tx, ty, tz = order[-1]
                    tx += nd[0]; ty += nd[1]; tz += nd[2]
                    if (tx, ty, tz) not in voxels:
                        d = nd
                        axes_used.add(self._axis_of(d))
                        break
                else:
                    break

            # Final acceptance test
            required_axes = {"x", "y", "z"}
            if len(voxels) == N and axes_used == required_axes:
                flipped = self._flip_voxels(order, axes=("x",))
                if not self._are_rotationally_equivalent(order, flipped):
                    return self._shift_to_origin(order)

        raise RuntimeError("Could not build a snake in the allotted attempts.")
    
    def _shift_to_origin(self, vox: List[Voxel]) -> List[Voxel]:
        """Shift voxel coordinates so the minimum is at origin."""
        if not vox:
            return []
        mins = [min(coord) for coord in zip(*vox)]
        dx, dy, dz = (-m for m in mins)
        return [(x + dx, y + dy, z + dz) for x, y, z in vox]

    def _orthogonals(self, d: Voxel) -> List[Voxel]:
        """Return the four directions that are perpendicular to `d`."""
        bad = {d, (-d[0], -d[1], -d[2])}
        return [v for v in DIRS if v not in bad]

    def _neighbour_count(self, v: Voxel, voxels: Set[Voxel]) -> int:
        """Number of occupied neighbours of voxel `v`."""
        x, y, z = v
        return sum((x + dx, y + dy, z + dz) in voxels for dx, dy, dz in DIRS)

    def _axis_of(self, d: Voxel) -> str:
        """Return 'x', 'y', or 'z' for a unit direction vector."""
        for idx, val in enumerate(d):
            if val != 0:
                return "xyz"[idx]
        raise ValueError("Invalid direction vector")

    def _flip_voxels(self, voxels: List[Voxel], axes: Tuple[str, ...] = ("x",)) -> List[Voxel]:
        """Flip voxels along specified axes."""
        if not voxels:
            return []

        xs, ys, zs = zip(*voxels)
        bounds = {
            "x": (min(xs), max(xs)),
            "y": (min(ys), max(ys)),
            "z": (min(zs), max(zs))
        }

        result = []
        for x, y, z in voxels:
            if "x" in axes:
                x = bounds["x"][1] - (x - bounds["x"][0])
            if "y" in axes:
                y = bounds["y"][1] - (y - bounds["y"][0])
            if "z" in axes:
                z = bounds["z"][1] - (z - bounds["z"][0])
            result.append((x, y, z))
        return result

    def _are_rotationally_equivalent(self, A: Iterable[Voxel], B: Iterable[Voxel]) -> bool:
        """Return True if set A can be rotated to match set B."""
        A, B = list(A), list(B)
        if len(A) != len(B):
            return False
            
        B_canon = self._canonicalize(B)
        rotations = self._rotation_matrices()
        
        for mat in rotations:
            A_rot = [self._apply_rotation(v, mat) for v in A]
            if self._canonicalize(A_rot) == B_canon:
                return True
        return False

    def _canonicalize(self, voxels: Iterable[Voxel]) -> Set[Voxel]:
        """Translate voxels so the minimal coordinate is at origin."""
        voxels = list(voxels)
        if not voxels:
            return set()
        anchor = min(voxels)
        ax, ay, az = anchor
        return {(x-ax, y-ay, z-az) for x, y, z in voxels}

    def _apply_rotation(self, v: Voxel, mat) -> Voxel:
        """Apply rotation matrix to a voxel."""
        x, y, z = v
        return (
            mat[0][0]*x + mat[0][1]*y + mat[0][2]*z,
            mat[1][0]*x + mat[1][1]*y + mat[1][2]*z,
            mat[2][0]*x + mat[2][1]*y + mat[2][2]*z,
        )

    def _rotation_matrices(self):
        """Generate the 24 orientation-preserving 3×3 rotation matrices."""
        mats = []
        for perm in permutations(range(3)):
            inversions = sum(perm[i] > perm[j] for i in range(3) for j in range(i+1, 3))
            parity = inversions % 2

            for signs in product((1, -1), repeat=3):
                det = signs[0] * signs[1] * signs[2] * (-1)**parity
                if det == 1:
                    mat = [[0]*3 for _ in range(3)]
                    for row, (axis, s) in enumerate(zip(perm, signs)):
                        mat[row][axis] = s
                    mats.append(tuple(tuple(r) for r in mat))
        return mats
    
    # ══════════════════════════════════════════════════════════════════════════
    #  3D RENDERING
    # ══════════════════════════════════════════════════════════════════════════
    
    def _render_voxel_image(self, voxels: List[Voxel], elev: float, azim: float) -> Image.Image:
        """Render voxel structure from specified viewing angle and return PIL Image."""
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, subplot_kw={'projection': '3d'})
        
        # Plot cubes with proper lighting and styling
        self._plot_cubes(voxels, ax, elev=elev, azim=azim)
        
        # Save to temporary buffer
        import io
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight', pad_inches=0.1, dpi=150, format='png')
        plt.close()
        
        # Process and resize to target size
        image = Image.open(buf)
        image = self._process_image(image, self.config.image_size)
        
        return image
    
    def _plot_cubes(
        self,
        positions: List[Voxel],
        ax,
        *,
        size: float = 1,
        elev: float = 25,
        azim: float = 35,
    ):
        """Draw cubes in 3D plot with proper styling."""
        # Generate cube faces with consistent coloring
        faces, colors = [], []
        for pos in positions:
            verts = self._cube_vertices(pos, size)
            for face in self._cube_faces(verts):
                faces.append(face)
                colors.append((0.7, 0.7, 0.9))  # Light blue color
        
        # Create 3D collection
        coll = Poly3DCollection(
            faces, 
            facecolors=colors, 
            linewidths=0.8, 
            edgecolors='black',
            alpha=0.8
        )
        ax.add_collection3d(coll)
        
        # Set up axes with proper scaling
        all_points = np.concatenate([self._cube_vertices(p, size) for p in positions])
        self._set_axes_equal(ax, all_points)
        
        # Configure view
        ax.set_proj_type('persp')
        ax.set_axis_off()
        ax.set_facecolor('white')
        ax.view_init(elev=elev, azim=azim)

    def _cube_vertices(self, origin: Voxel, size: float = 1) -> np.ndarray:
        """Generate vertices for a cube at given origin."""
        x, y, z = origin
        return np.array([
            [x, y, z], [x + size, y, z], [x + size, y + size, z], [x, y + size, z],
            [x, y, z + size], [x + size, y, z + size], [x + size, y + size, z + size], [x, y + size, z + size],
        ])

    def _cube_faces(self, verts: np.ndarray) -> List[List[np.ndarray]]:
        """Generate faces for a cube given its vertices."""
        return [
            [verts[j] for j in [0, 1, 2, 3]],  # bottom
            [verts[j] for j in [4, 5, 6, 7]],  # top
            [verts[j] for j in [0, 1, 5, 4]],  # front
            [verts[j] for j in [2, 3, 7, 6]],  # back
            [verts[j] for j in [1, 2, 6, 5]],  # right
            [verts[j] for j in [4, 7, 3, 0]],  # left
        ]

    def _set_axes_equal(self, ax, pts: np.ndarray, padding: float = 0.2):
        """Force equal scaling on all axes with padding."""
        max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2
        mid = pts.mean(axis=0)
        pad = max_range * padding
        
        for i, (set_lim, mid_val) in enumerate(zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid)):
            set_lim(mid_val - max_range - pad, mid_val + max_range + pad)
    
    def _process_image(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Process image: crop to square and resize to target size."""
        # Crop to square if needed
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        image = image.crop((left, top, left + size, top + size))
        
        # Resize to target size and convert to RGB
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image = image.convert("RGB")
        
        return image
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_video(
        self,
        first_image: Image.Image,
        final_image: Image.Image,
        task_id: str,
        voxels: List[Voxel],
        elev1: float,
        azim1: float,
        elev2: float,
        azim2: float
    ) -> Optional[str]:
        """Generate ground truth video showing smooth camera rotation."""
        temp_dir = Path(tempfile.gettempdir()) / f"{self.config.domain}_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{task_id}_ground_truth.mp4"
        
        # Generate intermediate frames for smooth rotation
        frames = self._create_rotation_frames(
            voxels, elev1, azim1, elev2, azim2,
            hold_frames=5,
            transition_frames=25
        )
        
        result = self.video_generator.create_video_from_frames(
            frames,
            video_path
        )
        
        return str(result) if result else None
    
    def _create_rotation_frames(
        self,
        voxels: List[Voxel],
        elev1: float,
        azim1: float,
        elev2: float,
        azim2: float,
        hold_frames: int = 5,
        transition_frames: int = 25
    ) -> List[Image.Image]:
        """Create animation frames showing smooth camera rotation."""
        frames = []
        
        # Hold initial position
        for _ in range(hold_frames):
            frames.append(self._render_voxel_image(voxels, elev1, azim1))
        
        # Create transition frames with smooth interpolation
        for i in range(transition_frames):
            progress = i / (transition_frames - 1) if transition_frames > 1 else 1.0
            
            # Interpolate azimuth (horizontal rotation)
            current_azim = azim1 + (azim2 - azim1) * progress
            # Keep elevation constant (horizontal rotation only)
            current_elev = elev1
            
            # Handle wrap-around for azimuth
            if abs(azim2 - azim1) > 180:
                if azim2 > azim1:
                    current_azim = azim1 - (360 - (azim2 - azim1)) * progress
                else:
                    current_azim = azim1 + (360 - (azim1 - azim2)) * progress
                current_azim = current_azim % 360
            
            frame = self._render_voxel_image(voxels, current_elev, current_azim)
            frames.append(frame)
        
        # Hold final position
        for _ in range(hold_frames):
            frames.append(self._render_voxel_image(voxels, elev2, azim2))
        
        return frames
