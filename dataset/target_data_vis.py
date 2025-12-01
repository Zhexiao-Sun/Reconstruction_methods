import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import subprocess
from typing import Tuple, List, Optional
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import math

class TargetVisualizer:
    def __init__(self, dataset_path: str = "Target_trajectories"):
        """
        Initialize the target dataset visualizer.
        
        Args:
            dataset_path: Path to the Target_trajectories folder
        """
        self.dataset_path = Path(dataset_path)
        self.data_path = self.dataset_path / "data"
        self.video_path = self.dataset_path / "video"
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    @staticmethod
    def quaternion_to_heading(qx, qy, qz, qw):
        """
        Convert quaternion to heading angle in radians.
        
        Args:
            qx, qy, qz, qw: Quaternion components
            
        Returns:
            Heading angle in radians (0 to 2π)
        """
        # Convert quaternion to yaw angle (rotation around z-axis)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 
                        1.0 - 2.0 * (qy * qy + qz * qz))
        
        # Normalize to [0, 2π]
        if yaw < 0:
            yaw += 2 * math.pi
            
        return yaw
    
    @staticmethod
    def get_heading_vector(qx, qy, qz, qw, length=1.0):
        """
        Get heading direction vector from quaternion.
        
        Args:
            qx, qy, qz, qw: Quaternion components
            length: Length of the direction vector
            
        Returns:
            Tuple of (dx, dy) representing the direction vector
        """
        heading = TargetVisualizer.quaternion_to_heading(qx, qy, qz, qw)
        dx = length * math.cos(heading)
        dy = length * math.sin(heading)
        return dx, dy
    
    @staticmethod
    def rotate_points(x_coords, y_coords, angle_rad):
        """
        Rotate a set of 2D points by a given angle.
        
        Args:
            x_coords: Array of x coordinates
            y_coords: Array of y coordinates  
            angle_rad: Rotation angle in radians
            
        Returns:
            Tuple of (rotated_x, rotated_y) arrays
        """
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        rotated_x = x_coords * cos_angle - y_coords * sin_angle
        rotated_y = x_coords * sin_angle + y_coords * cos_angle
        
        return rotated_x, rotated_y
    
    @staticmethod
    def rotate_quaternion_heading(qx, qy, qz, qw, rotation_angle):
        """
        Rotate a quaternion's heading by a given angle around the z-axis.
        
        Args:
            qx, qy, qz, qw: Original quaternion components
            rotation_angle: Additional rotation angle in radians
            
        Returns:
            Tuple of (new_qx, new_qy, new_qz, new_qw) representing rotated quaternion
        """
        # Create a quaternion for the additional rotation around z-axis
        half_angle = rotation_angle / 2
        rot_qx = 0
        rot_qy = 0
        rot_qz = math.sin(half_angle)
        rot_qw = math.cos(half_angle)
        
        # Multiply quaternions: q_new = q_rotation * q_original
        new_qw = rot_qw * qw - rot_qx * qx - rot_qy * qy - rot_qz * qz
        new_qx = rot_qw * qx + rot_qx * qw + rot_qy * qz - rot_qz * qy
        new_qy = rot_qw * qy - rot_qx * qz + rot_qy * qw + rot_qz * qx
        new_qz = rot_qw * qz + rot_qx * qy - rot_qy * qx + rot_qz * qw
        
        return new_qx, new_qy, new_qz, new_qw
    
    def load_scene_data(self, scene_name: str) -> Tuple[None, pd.DataFrame]:
        """
        Load trajectory data for a specific scene (no annotations).
        
        Args:
            scene_name: Name of the scene (e.g., "sample_096")
        
        Returns:
            Tuple of (None, trajectory_df) - annotations not used
        """
        # Load trajectory data only
        trajectory_file = self.data_path / f"{scene_name}.parquet"
        trajectory_df = pd.read_parquet(trajectory_file)
        
        return None, trajectory_df
    
    def extract_video_clip_frames(self, scene_name: str, start_frame: int, end_frame: int, 
                                 output_path: str = "temp_clip.mp4", output_fps: int = 5) -> str:
        """
        Extract a video clip using the same frame mapping logic as the GUI display.
        This ensures the saved video has the exact same frames as shown in the viewer.
        
        Args:
            scene_name: Name of the scene
            start_frame: Starting frame number (trajectory frame)
            end_frame: Ending frame number (trajectory frame)
            output_path: Output video file path
            output_fps: Frames per second for output video
        
        Returns:
            Path to the extracted video clip
        """
        video_file = self.video_path / f"{scene_name}_rgb_camera.mp4"
        
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        # Load trajectory data to get the segment
        trajectory_file = self.data_path / f"{scene_name}.parquet"
        if not trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
        
        trajectory_df = pd.read_parquet(trajectory_file)
        
        # Get trajectory segment (same as GUI)
        trajectory_segment = trajectory_df[
            (trajectory_df['frame'] >= start_frame) & 
            (trajectory_df['frame'] <= end_frame)
        ].copy()
        
        if len(trajectory_segment) == 0:
            raise ValueError("No trajectory data for the specified frame range")
        
        trajectory_frames = trajectory_segment['frame'].values
        
        # Open video and get properties
        cap = cv2.VideoCapture(str(video_file))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if video_fps <= 0:
            video_fps = 30  # Fallback FPS
        
        # Use the same frame mapping logic as the GUI
        traj_start = trajectory_frames[0]
        traj_end = trajectory_frames[-1]
        max_traj_frame = trajectory_df['frame'].max()
        
        # Map to video frames proportionally
        video_start_frame = int((traj_start / max_traj_frame) * total_video_frames)
        video_end_frame = int((traj_end / max_traj_frame) * total_video_frames)
        
        # Ensure bounds
        video_start_frame = max(0, min(video_start_frame, total_video_frames - 1))
        video_end_frame = max(video_start_frame + 1, min(video_end_frame, total_video_frames - 1))
        
        # Calculate frame step for uniform sampling
        if len(trajectory_frames) > 1:
            video_frame_span = video_end_frame - video_start_frame
            frame_step = video_frame_span / (len(trajectory_frames) - 1)
        else:
            frame_step = 0
        
        # Extract frames exactly as done in GUI
        extracted_frames = []
        for i, traj_frame in enumerate(trajectory_frames):
            if len(trajectory_frames) > 1:
                video_frame_pos = int(video_start_frame + i * frame_step)
            else:
                video_frame_pos = video_start_frame
            
            video_frame_pos = max(0, min(video_frame_pos, total_video_frames - 1))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_pos)
            ret, frame = cap.read()
            
            if ret and frame is not None and frame.size > 0:
                extracted_frames.append(frame)
            elif extracted_frames:
                # Duplicate last frame if current frame fails
                extracted_frames.append(extracted_frames[-1])
            else:
                # Create black placeholder if no frames yet
                placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                extracted_frames.append(placeholder_frame)
        
        cap.release()
        
        if not extracted_frames:
            raise ValueError("No frames could be extracted")
        
        # Write video using cv2.VideoWriter
        height, width = extracted_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        
        for frame in extracted_frames:
            out.write(frame)
        
        out.release()
        
        print(f"Extracted {len(extracted_frames)} frames to {output_path}")
        return output_path
    
    def transform_trajectory(self, trajectory_df: pd.DataFrame, start_frame: int, 
                           end_frame: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Transform trajectory coordinates to start from (0,0) with initial heading aligned to Y-axis.
        
        Args:
            trajectory_df: DataFrame containing trajectory data
            start_frame: Starting frame number
            end_frame: Ending frame number
        
        Returns:
            Tuple of (x_coords, y_coords, segment_df) where segment_df contains transformed quaternion data
        """
        # Filter trajectory data for the specified frame range
        segment_df = trajectory_df[
            (trajectory_df['frame'] >= start_frame) & 
            (trajectory_df['frame'] <= end_frame)
        ].copy()
        
        if len(segment_df) == 0:
            return np.array([]), np.array([]), pd.DataFrame()
        
        # Extract x, y coordinates - using pos_x, pos_y instead of cart_x, cart_y
        x_coords = segment_df['pos_x'].values
        y_coords = segment_df['pos_y'].values
        
        # Transform to start from (0, 0)
        x_coords = x_coords - x_coords[0]
        y_coords = y_coords - y_coords[0]
        
        # Get initial heading from the first quaternion
        first_row = segment_df.iloc[0]
        initial_qx, initial_qy, initial_qz, initial_qw = (
            first_row['quat_x'], first_row['quat_y'], first_row['quat_z'], first_row['quat_w'])
        
        # Calculate the rotation needed to align initial heading with Y-axis
        initial_heading = self.quaternion_to_heading(initial_qx, initial_qy, initial_qz, initial_qw)
        # We want to rotate so that initial heading points to Y-axis (90 degrees or π/2 radians)
        rotation_angle = (math.pi / 2) - initial_heading
        
        # Apply rotation to all trajectory points
        rotated_x, rotated_y = self.rotate_points(x_coords, y_coords, rotation_angle)
        
        # Update the segment_df with transformed coordinates and rotated quaternions
        segment_df = segment_df.copy()
        segment_df['transformed_x'] = rotated_x
        segment_df['transformed_y'] = rotated_y
        
        # Apply the same rotation to all quaternions in the segment
        rotated_quaternions = []
        for _, row in segment_df.iterrows():
            qx, qy, qz, qw = row['quat_x'], row['quat_y'], row['quat_z'], row['quat_w']
            new_qx, new_qy, new_qz, new_qw = self.rotate_quaternion_heading(qx, qy, qz, qw, rotation_angle)
            rotated_quaternions.append((new_qx, new_qy, new_qz, new_qw))
        
        # Update quaternion columns with rotated values
        rotated_quats = np.array(rotated_quaternions)
        segment_df['transformed_quat_x'] = rotated_quats[:, 0]
        segment_df['transformed_quat_y'] = rotated_quats[:, 1]
        segment_df['transformed_quat_z'] = rotated_quats[:, 2]
        segment_df['transformed_quat_w'] = rotated_quats[:, 3]
        
        # Store the rotation angle used for this segment
        segment_df['rotation_angle'] = rotation_angle
        
        return rotated_x, rotated_y, segment_df
    
    def transform_trajectory_with_reference_frame(self, trajectory_df: pd.DataFrame, 
                                                start_frame: int, end_frame: int, 
                                                reference_frame: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Transform trajectory segment with a specific reference frame as the origin and heading alignment point.
        
        Args:
            trajectory_df: Full trajectory DataFrame
            start_frame: Start frame of the segment to extract
            end_frame: End frame of the segment to extract  
            reference_frame: Frame to use as the transformation reference (origin and heading alignment)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, pd.DataFrame]: x_coords, y_coords, segment_df
        """
        # Extract the segment
        segment_df = trajectory_df[
            (trajectory_df['frame'] >= start_frame) & 
            (trajectory_df['frame'] <= end_frame)
        ].copy()
        
        if len(segment_df) == 0:
            return np.array([]), np.array([]), pd.DataFrame()
        
        # Find the reference point within the segment
        reference_rows = segment_df[segment_df['frame'] == reference_frame]
        if len(reference_rows) == 0:
            # If reference frame is not in segment, use the closest frame
            closest_idx = np.argmin(np.abs(segment_df['frame'] - reference_frame))
            reference_row = segment_df.iloc[closest_idx]
            print(f"Reference frame {reference_frame} not found in segment, using closest frame {reference_row['frame']}")
        else:
            reference_row = reference_rows.iloc[0]
        
        # Get reference position and heading - using pos_x, pos_y instead of cart_x, cart_y
        ref_x = reference_row['pos_x'] 
        ref_y = reference_row['pos_y']
        ref_qx = reference_row['quat_x']
        ref_qy = reference_row['quat_y'] 
        ref_qz = reference_row['quat_z']
        ref_qw = reference_row['quat_w']
        
        # Calculate reference heading
        reference_heading = self.quaternion_to_heading(ref_qx, ref_qy, ref_qz, ref_qw)
        
        # Extract x, y coordinates for all points in segment - using pos_x, pos_y
        x_coords = segment_df['pos_x'].values
        y_coords = segment_df['pos_y'].values
        
        # Translate all points so reference point becomes origin
        x_coords = x_coords - ref_x
        y_coords = y_coords - ref_y
        
        # Calculate rotation needed to align reference heading with Y-axis (90 degrees)
        rotation_angle = math.pi / 2 - reference_heading
        
        # Apply rotation to all points
        rotated_x, rotated_y = self.rotate_points(x_coords, y_coords, rotation_angle)
        
        # Update the segment_df with transformed coordinates
        segment_df = segment_df.copy()
        segment_df['transformed_x'] = rotated_x
        segment_df['transformed_y'] = rotated_y
        
        # Apply same rotation to all quaternions in the segment
        rotated_quaternions = []
        for _, row in segment_df.iterrows():
            qx, qy, qz, qw = row['quat_x'], row['quat_y'], row['quat_z'], row['quat_w']
            new_qx, new_qy, new_qz, new_qw = self.rotate_quaternion_heading(qx, qy, qz, qw, rotation_angle)
            rotated_quaternions.append((new_qx, new_qy, new_qz, new_qw))
        
        # Update quaternion columns with rotated values
        rotated_quats = np.array(rotated_quaternions)
        segment_df['transformed_quat_x'] = rotated_quats[:, 0]
        segment_df['transformed_quat_y'] = rotated_quats[:, 1]
        segment_df['transformed_quat_z'] = rotated_quats[:, 2]
        segment_df['transformed_quat_w'] = rotated_quats[:, 3]
        
        # Store the rotation angle and reference frame used for this segment
        segment_df['rotation_angle'] = rotation_angle
        segment_df['reference_frame'] = reference_frame
        
        return rotated_x, rotated_y, segment_df
    
    def list_available_scenes(self) -> List[str]:
        """
        List all available scenes in the dataset by scanning data directory.
        
        Returns:
            List of scene names
        """
        scenes = []
        for file in self.data_path.glob("*.parquet"):
            scene_name = file.stem  # e.g., "sample_096.parquet" -> "sample_096"
            scenes.append(scene_name)
        return sorted(scenes)


def main():
    """
    Main function demonstrating the usage of TargetVisualizer.
    """
    # Initialize visualizer
    visualizer = TargetVisualizer()
    
    # List available scenes
    scenes = visualizer.list_available_scenes()
    print(f"Available scenes: {len(scenes)}")
    print("First 5 scenes:", scenes[:5])
    
    # Visualize a specific scene segment
    if scenes:
        scene_name = scenes[0]  # Use the first available scene
        print(f"\nVisualizing scene: {scene_name}")
        
        # Load and display scene info
        _, trajectory_df = visualizer.load_scene_data(scene_name)
        print(f"Scene has {len(trajectory_df)} trajectory points")


if __name__ == "__main__":
    main()

