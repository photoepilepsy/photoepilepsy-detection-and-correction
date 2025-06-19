import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedVideoFlashCorrector:
    """
    Advanced video flash correction system that handles photosensitive content
    with sophisticated color filtering and temporal smoothing.
    """
    
    def __init__(self, working_folder: str, json_filename: str = "results.json", output_suffix: str = "_corrected"):
        """
        Initialize the video flash corrector.
        
        Args:
            working_folder: Name of the subfolder containing video and JSON files
            json_filename: Name of the JSON file (default: "results.json")
            output_suffix: Suffix to add to the output video filename
        """
        self.working_folder = working_folder
        self.json_filename = json_filename
        self.output_suffix = output_suffix
        
        # Construct full paths
        self.json_path = os.path.join(working_folder, json_filename)
        
        # Validate working folder exists
        if not os.path.exists(working_folder):
            raise FileNotFoundError(f"Working folder not found: {working_folder}")
        
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        
        # Load prediction data and extract video path
        self.prediction_data = self._load_json_data()
        self.video_path = self._get_video_path()
        self.output_path = self._generate_output_path()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Advanced correction parameters
        self.flash_threshold = 180  # Brightness threshold for flash detection
        self.color_intensity_threshold = 200  # Threshold for intense colors
        self.temporal_window_size = 7  # Frames for temporal smoothing
        self.overlay_opacity = 0.3  # Opacity of the protective overlay
        self.gaussian_sigma = 1.0  # Gaussian blur sigma for smoothing
        
        # Flickering detection parameters
        self.flicker_threshold = 30  # Minimum brightness change to detect flicker
        self.flicker_frequency_threshold = 3  # Hz - dangerous flicker frequency
        self.flicker_smoothing_strength = 0.6  # Strength of anti-flicker smoothing
        
        logger.info(f"Video loaded: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        logger.info(f"Output will be saved to: {self.output_path}")
    
    def _load_json_data(self) -> Dict[str, Any]:
        """Load and validate JSON prediction data."""
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            if 'predictions' not in data or 'video_info' not in data:
                raise ValueError("JSON file must contain 'predictions' and 'video_info' fields")
            
            logger.info(f"Loaded {len(data['predictions'])} prediction windows")
            return data
        
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            raise
    
    def _get_video_path(self) -> str:
        """Extract video path from JSON data and construct full path within working folder."""
        filename = self.prediction_data['video_info']['filename']
        
        # Construct full path within working folder
        full_path = os.path.join(self.working_folder, filename)
        
        if os.path.exists(full_path):
            return full_path
        
        raise FileNotFoundError(f"Video file not found: {full_path} (filename from JSON: {filename})")
    
    def _generate_output_path(self) -> str:
        """Generate output video path with suffix within the working folder."""
        video_name = os.path.basename(self.video_path)
        name_without_ext, ext = os.path.splitext(video_name)
        
        output_name = f"{name_without_ext}{self.output_suffix}{ext}"
        return os.path.join(self.working_folder, output_name)
    
    def _detect_flickering(self, frames: List[np.ndarray]) -> Tuple[bool, List[float], np.ndarray]:
        """
        Detect flickering patterns in frame sequence.
        
        Args:
            frames: List of frames to analyze
            
        Returns:
            Tuple of (has_flickering, brightness_values, flicker_mask)
        """
        if len(frames) < 3:
            return False, [], np.zeros((self.height, self.width), dtype=np.float32)
        
        brightness_values = []
        frame_grays = []
        
        # Calculate brightness for each frame
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_grays.append(gray)
            brightness_values.append(np.mean(gray))
        
        # Detect rapid brightness changes
        brightness_changes = np.abs(np.diff(brightness_values))
        has_rapid_changes = np.any(brightness_changes > self.flicker_threshold)
        
        # Calculate per-pixel flicker intensity
        flicker_mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        if len(frame_grays) >= 3:
            # Calculate variance across frames for each pixel
            stacked_grays = np.stack(frame_grays, axis=0)
            pixel_variance = np.var(stacked_grays, axis=0)
            
            # Normalize variance to 0-1 range
            max_variance = np.max(pixel_variance)
            if max_variance > 0:
                flicker_mask = pixel_variance / max_variance
                
                # Apply threshold to focus on high-variance areas
                flicker_mask = np.where(flicker_mask > 0.3, flicker_mask, 0)
        
        # Check for dangerous flicker frequencies
        if len(brightness_values) >= 5:
            # Simple frequency analysis using differences
            changes_per_second = np.sum(brightness_changes > self.flicker_threshold) * self.fps / len(frames)
            has_dangerous_frequency = changes_per_second >= self.flicker_frequency_threshold
        else:
            has_dangerous_frequency = False
        
        has_flickering = has_rapid_changes or has_dangerous_frequency
        
    def _detect_intense_colors(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detect intense red/blue colors and create a mask.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple of (has_intense_colors, color_mask)
        """
        # Convert to float for calculations
        frame_float = frame.astype(np.float32)
        
        # Extract color channels (BGR format)
        blue_channel = frame_float[:, :, 0]
        green_channel = frame_float[:, :, 1]
        red_channel = frame_float[:, :, 2]
        
        # Detect intense reds and blues
        intense_red_mask = (red_channel > self.color_intensity_threshold) & \
                          (red_channel > blue_channel * 1.5) & \
                          (red_channel > green_channel * 1.5)
        
        intense_blue_mask = (blue_channel > self.color_intensity_threshold) & \
                           (blue_channel > red_channel * 1.5) & \
                           (blue_channel > green_channel * 1.5)
        
        # Combine masks
        combined_mask = intense_red_mask | intense_blue_mask
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Blur the mask for smoother transitions
        combined_mask = gaussian_filter(combined_mask.astype(np.float32), sigma=2.0)
        
        has_intense_colors = np.any(combined_mask > 0.1)
        
        return has_intense_colors, combined_mask
    
    def _calculate_frame_metrics(self, frame: np.ndarray) -> Dict[str, float]:
        """Calculate various metrics for adaptive correction."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        metrics = {
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'red_intensity': np.mean(frame[:, :, 2]),
            'blue_intensity': np.mean(frame[:, :, 0]),
            'saturation': np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1])
        }
        
        return metrics
    
    def _create_adaptive_overlay(self, frame: np.ndarray, metrics: Dict[str, float], 
                               color_mask: np.ndarray) -> np.ndarray:
        """
        Create an adaptive overlay based on frame characteristics.
        
        Args:
            frame: Input frame
            metrics: Frame metrics
            color_mask: Mask of intense colors
            
        Returns:
            Overlay to be applied to the frame
        """
        overlay = np.zeros_like(frame, dtype=np.float32)
        
        # Base overlay intensity based on brightness
        if metrics['brightness'] > self.flash_threshold:
            base_intensity = min(0.6, (metrics['brightness'] - self.flash_threshold) / 100)
        else:
            base_intensity = 0.2
        
        # Adjust for high contrast (likely flash)
        if metrics['contrast'] > 80:
            base_intensity += 0.2
        
        # Create gradient overlay that's stronger in areas with intense colors
        for i in range(3):  # For each color channel
            channel_overlay = np.full_like(frame[:, :, i], base_intensity * 255, dtype=np.float32)
            
            # Increase overlay intensity where there are intense colors
            intense_areas = color_mask > 0.1
            channel_overlay[intense_areas] = np.minimum(
                channel_overlay[intense_areas] + color_mask[intense_areas] * 100,
                255 * 0.7  # Cap at 70% overlay
            )
            
            overlay[:, :, i] = channel_overlay
        
        # Apply slight warm tint to counteract harsh flashes
        overlay[:, :, 0] *= 0.9  # Reduce blue slightly
        overlay[:, :, 1] *= 0.95  # Reduce green slightly
        overlay[:, :, 2] *= 1.0   # Keep red
        
        return overlay
    
    def _apply_temporal_smoothing(self, frames: List[np.ndarray], flicker_mask: np.ndarray = None) -> List[np.ndarray]:
        """Apply temporal smoothing across multiple frames with anti-flicker enhancement."""
        if len(frames) < 3:
            return frames
        
        smoothed_frames = []
        
        for i in range(len(frames)):
            # Define window around current frame
            start_idx = max(0, i - self.temporal_window_size // 2)
            end_idx = min(len(frames), i + self.temporal_window_size // 2 + 1)
            
            # Calculate weights (current frame has more weight, but less for flickering areas)
            base_weights = np.exp(-np.abs(np.arange(start_idx, end_idx) - i) * 0.3)
            
            smoothed_frame = np.zeros_like(frames[i], dtype=np.float32)
            
            # Apply different smoothing strategies
            if flicker_mask is not None and np.any(flicker_mask > 0.1):
                # Enhanced smoothing for flickering areas
                for j, idx in enumerate(range(start_idx, end_idx)):
                    weight = base_weights[j]
                    frame_contribution = frames[idx].astype(np.float32)
                    
                    # Reduce weight of current frame in flickering areas
                    if idx == i:
                        flicker_areas = flicker_mask > 0.1
                        current_weight = np.ones_like(flicker_mask)
                        current_weight[flicker_areas] *= (1 - self.flicker_smoothing_strength)
                        
                        for c in range(3):  # For each color channel
                            frame_contribution[:, :, c] *= current_weight
                    
                    smoothed_frame += frame_contribution * weight
                
                # Normalize by total weights
                total_weight = np.sum(base_weights)
                smoothed_frame /= total_weight
                
                # Apply additional low-pass filtering to flickering areas
                flicker_areas_3d = np.stack([flicker_mask] * 3, axis=2) > 0.2
                
                # Gaussian blur for flickering areas
                blurred_frame = np.zeros_like(smoothed_frame)
                for c in range(3):
                    blurred_frame[:, :, c] = gaussian_filter(smoothed_frame[:, :, c], sigma=1.5)
                
                # Blend original and blurred based on flicker intensity
                blend_factor = np.stack([flicker_mask] * 3, axis=2) * 0.7
                smoothed_frame = smoothed_frame * (1 - blend_factor) + blurred_frame * blend_factor
                
            else:
                # Standard temporal smoothing
                weights = base_weights / np.sum(base_weights)
                
                for j, idx in enumerate(range(start_idx, end_idx)):
                    smoothed_frame += frames[idx].astype(np.float32) * weights[j]
            
            smoothed_frames.append(np.clip(smoothed_frame, 0, 255).astype(np.uint8))
        
        return smoothed_frames
    
    def _correct_frame_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply comprehensive correction to a sequence of frames with enhanced flicker detection.
        
        Args:
            frames: List of frames to correct
            
        Returns:
            List of corrected frames
        """
        if not frames:
            return frames
        
        # Detect flickering patterns in the sequence
        has_flickering, brightness_values, flicker_mask = self._detect_flickering(frames)
        
        corrected_frames = []
        
        for i, frame in enumerate(frames):
            # Calculate frame metrics
            metrics = self._calculate_frame_metrics(frame)
            
            # Detect intense colors
            has_intense_colors, color_mask = self._detect_intense_colors(frame)
            
            # Convert frame to float for processing
            frame_float = frame.astype(np.float32)
            
            # Determine if correction is needed
            needs_correction = (has_intense_colors or 
                              metrics['brightness'] > self.flash_threshold or
                              has_flickering)
            
            if needs_correction:
                # Create adaptive overlay
                overlay = self._create_adaptive_overlay(frame, metrics, color_mask)
                
                # Apply overlay with variable opacity
                opacity = self.overlay_opacity
                
                # Increase opacity for flickering content
                if has_flickering:
                    flicker_intensity = np.mean(flicker_mask) if flicker_mask is not None else 0
                    opacity += flicker_intensity * 0.3
                
                if metrics['brightness'] > 220:  # Very bright flash
                    opacity = min(0.7, opacity + 0.2)
                
                # Blend frame with overlay
                corrected_frame = frame_float * (1 - opacity) + overlay * opacity
                
                # Apply additional color correction for intense areas
                if has_intense_colors:
                    # Desaturate intense color areas
                    hsv = cv2.cvtColor(corrected_frame.astype(np.uint8), cv2.COLOR_BGR2HSV)
                    hsv = hsv.astype(np.float32)
                    
                    # Reduce saturation in intense areas
                    saturation_reduction = color_mask * 0.5
                    hsv[:, :, 1] *= (1 - saturation_reduction)
                    
                    corrected_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
                
                # Apply flickering-specific corrections
                if has_flickering and flicker_mask is not None:
                    # Apply stronger blur to flickering areas
                    blurred_frame = gaussian_filter(corrected_frame, sigma=2.0)
                    
                    # Blend based on flicker intensity
                    flicker_blend = np.stack([flicker_mask] * 3, axis=2) * 0.8
                    corrected_frame = corrected_frame * (1 - flicker_blend) + blurred_frame * flicker_blend
                
                # Apply slight gaussian blur to harsh transitions
                corrected_frame = gaussian_filter(corrected_frame, sigma=self.gaussian_sigma)
                
            else:
                corrected_frame = frame_float
            
            corrected_frames.append(np.clip(corrected_frame, 0, 255).astype(np.uint8))
        
        # Apply enhanced temporal smoothing with flicker detection
        if len(corrected_frames) > 2:
            corrected_frames = self._apply_temporal_smoothing(corrected_frames, flicker_mask if has_flickering else None)
        
        return corrected_frames
    
    def _get_correction_segments(self) -> List[Tuple[int, int]]:
        """Get frame segments that need correction based on predictions."""
        segments = []
        current_start = None
        
        for prediction in self.prediction_data['predictions']:
            if prediction['prediction'] == 1:  # Needs correction
                if current_start is None:
                    current_start = prediction['start_frame']
                current_end = prediction['end_frame']
            else:
                if current_start is not None:
                    segments.append((current_start, current_end))
                    current_start = None
        
        # Handle case where correction needed till the end
        if current_start is not None:
            segments.append((current_start, current_end))
        
        logger.info(f"Found {len(segments)} segments requiring correction")
        for i, (start, end) in enumerate(segments):
            logger.info(f"Segment {i+1}: frames {start}-{end} ({end-start} frames)")
        
        return segments
    
    def process_video(self) -> None:
        """Process the entire video with advanced flash correction."""
        logger.info("Starting advanced video processing...")
        
        # Get correction segments
        correction_segments = self._get_correction_segments()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        try:
            current_frame = 0
            total_corrected_frames = 0
            
            for segment_idx, (start_frame, end_frame) in enumerate(correction_segments):
                # Write unchanged frames before correction segment
                if current_frame < start_frame:
                    frames_written = self._write_unchanged_frames(out, current_frame, start_frame)
                    logger.info(f"Wrote {frames_written} unchanged frames ({current_frame}-{start_frame})")
                
                # Process correction segment
                segment_size = end_frame - start_frame
                logger.info(f"Processing segment {segment_idx + 1}/{len(correction_segments)}: "
                           f"frames {start_frame}-{end_frame} ({segment_size} frames)")
                
                # Process in chunks for memory efficiency (smaller chunks for better flicker detection)
                chunk_size = min(90, segment_size)  # Smaller chunks for better temporal analysis
                
                for chunk_start in range(start_frame, end_frame, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, end_frame)
                    
                    # Read chunk of frames
                    frames = self._read_frame_chunk(chunk_start, chunk_end)
                    
                    if frames:
                        # Apply correction
                        corrected_frames = self._correct_frame_sequence(frames)
                        
                        # Write corrected frames
                        for frame in corrected_frames:
                            out.write(frame)
                        
                        total_corrected_frames += len(corrected_frames)
                        
                        # Progress update
                        progress = (chunk_end - start_frame) / segment_size * 100
                        logger.info(f"Segment {segment_idx + 1} progress: {progress:.1f}%")
                
                current_frame = end_frame
            
            # Write remaining unchanged frames
            if current_frame < self.total_frames:
                frames_written = self._write_unchanged_frames(out, current_frame, self.total_frames)
                logger.info(f"Wrote final {frames_written} unchanged frames")
            
            logger.info(f"Processing completed! Total corrected frames: {total_corrected_frames}")
            
        finally:
            out.release()
            self.cap.release()
        
        logger.info(f"Video saved to: {self.output_path}")
    
    def _read_frame_chunk(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """Read a chunk of frames efficiently."""
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                break
            frames.append(frame)
        
        return frames
    
    def _write_unchanged_frames(self, writer: cv2.VideoWriter, start_frame: int, end_frame: int) -> int:
        """Write unchanged frames to output video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_written = 0
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                break
            writer.write(frame)
            frames_written += 1
        
        return frames_written
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        correction_segments = self._get_correction_segments()
        
        total_corrected_frames = sum(end - start for start, end in correction_segments)
        correction_percentage = (total_corrected_frames / self.total_frames) * 100
        
        return {
            "input_video": self.video_path,
            "output_video": self.output_path,
            "total_frames": self.total_frames,
            "corrected_frames": total_corrected_frames,
            "unchanged_frames": self.total_frames - total_corrected_frames,
            "correction_percentage": round(correction_percentage, 2),
            "correction_segments": len(correction_segments),
            "video_duration_seconds": round(self.total_frames / self.fps, 2),
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}"
        }


def main():
    """Main function to run the advanced video flash correction."""
    
    # Configuration parameters
    working_folder = "video1"  # Name of the subfolder containing video and JSON files
    json_filename = "results.json"  # Name of the JSON file within the working folder
    output_suffix = "_flash_corrected"  # Suffix for the output video file
    
    try:
        # Initialize corrector
        logger.info(f"Initializing Advanced Video Flash Corrector for folder: {working_folder}")
        corrector = AdvancedVideoFlashCorrector(working_folder, json_filename, output_suffix)
        
        # Print processing statistics
        stats = corrector.get_processing_stats()
        logger.info("Processing Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Process the video
        corrector.process_video()
        
        logger.info("Advanced flash correction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
