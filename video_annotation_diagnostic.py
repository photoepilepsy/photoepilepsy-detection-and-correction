import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
from typing import List, Dict, Tuple
import seaborn as sns

class VideoAnnotationDiagnostic:
    def __init__(self, dataset_type: str = 'val'):
        """
        Initialize the diagnostic tool with dataset type
        
        Args:
            dataset_type: Type of dataset ('val', 'test', or 'train')
        """
        self.dataset_type = dataset_type
        self.json1_path = f'analytic/{dataset_type}/{dataset_type}_set.json'
        self.json2_path = f'analytic/{dataset_type}/{dataset_type}_set_peat.json'
        self.video_path = f'analytic/{dataset_type}/{dataset_type}_set.mp4'
        
        # Load JSON data
        self.data1 = self.load_json(self.json1_path)
        self.data2 = self.load_json(self.json2_path)
        
        # Get video properties
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Dataset: {self.dataset_type}")
        print(f"JSON 1: {self.json1_path}")
        print(f"JSON 2: {self.json2_path}")
        print(f"Video: {self.video_path}")
        print(f"Video FPS: {self.fps}")
        print(f"Total frames: {self.total_frames}")
        
    def load_json(self, path: str) -> List[Dict]:
        """Load JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def create_frame_annotations(self, data: List[Dict]) -> np.ndarray:
        """
        Create frame-by-frame annotations array
        
        Args:
            data: JSON data with segments
            
        Returns:
            numpy array where each index represents frame number and value is effect (0 or 1)
            Uses 0 for unannotated frames
        """
        annotations = np.zeros(self.total_frames, dtype=int)  # 0 for unannotated
        
        for video_data in data:
            for segment in video_data['segments']:
                start_frame = segment['start_frame']
                end_frame = segment['end_frame']
                effect = segment['effect']
                
                # Ensure we don't go beyond video length
                end_frame = min(end_frame, self.total_frames)
                
                annotations[start_frame:end_frame] = effect
                
        return annotations
    
    def find_differences(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Find frames where the two JSON files differ
        Special logic: frames not in peat conversion are marked as 0
        
        Returns:
            Tuple of (difference_mask, difference_ranges)
        """
        annotations1 = self.create_frame_annotations(self.data1)
        annotations2 = self.create_frame_annotations(self.data2)
        
        # Track which frames are actually annotated in the original JSON
        annotated1 = np.zeros(self.total_frames, dtype=bool)
        annotated2 = np.zeros(self.total_frames, dtype=bool)
        
        # Mark annotated frames for original JSON
        for video_data in self.data1:
            for segment in video_data['segments']:
                start_frame = segment['start_frame']
                end_frame = min(segment['end_frame'], self.total_frames)
                annotated1[start_frame:end_frame] = True
                
        # Mark annotated frames for peat JSON
        for video_data in self.data2:
            for segment in video_data['segments']:
                start_frame = segment['start_frame']
                end_frame = min(segment['end_frame'], self.total_frames)
                annotated2[start_frame:end_frame] = True
        
        # SPECIAL LOGIC: For frames that are in original but NOT in peat, 
        # set peat annotation to 0 (meaning "not converted")
        frames_in_original_not_peat = annotated1 & ~annotated2
        annotations2[frames_in_original_not_peat] = 0
        
        # Now compare all frames that are in the original annotation
        # (either annotated in peat OR set to 0 because not in peat)
        valid_mask = annotated1
        difference_mask = valid_mask & (annotations1 != annotations2)
        
        # Find continuous ranges of differences
        diff_frames = np.where(difference_mask)[0]
        difference_ranges = []
        
        if len(diff_frames) > 0:
            start = diff_frames[0]
            prev = diff_frames[0]
            
            for frame in diff_frames[1:]:
                if frame - prev > 1:  # Gap found
                    difference_ranges.append((start, prev))
                    start = frame
                prev = frame
            
            # Add the last range
            difference_ranges.append((start, prev))
        
        return difference_mask, difference_ranges, annotations1, annotations2
    
    def extract_frames_from_range(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """
        Extract ALL frames from a specific range
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            
        Returns:
            List of all frame images in the range
        """
        frames = []
        
        for frame_num in range(start_frame, end_frame + 1):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_num, frame_rgb))
                
        return frames
    
    def draw_text_with_outline(self, draw, position, text, font, fill_color, outline_color, outline_width=3):
        """
        Draw text with outline for better visibility
        """
        x, y = position
        # Draw outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        # Draw main text
        draw.text(position, text, font=font, fill=fill_color)
    
    def create_comparison_image(self, frame_num: int, frame: np.ndarray, 
                               effect1: int, effect2: int) -> Image.Image:
        """
        Create a comparison image with annotations
        
        Args:
            frame_num: Frame number
            frame: Frame image
            effect1: Effect value from first JSON
            effect2: Effect value from second JSON
            
        Returns:
            PIL Image with annotations
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a larger, bolder font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 40)
            small_font = ImageFont.truetype("arial.ttf", 32)
        except:
            try:
                font = ImageFont.truetype("ArialBold.ttf", 40)
                small_font = ImageFont.truetype("ArialBold.ttf", 32)
            except:
                try:
                    # Try common system fonts
                    font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 40)
                    small_font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 32)
                except:
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
        
        # Add frame number with black outline for better visibility
        self.draw_text_with_outline(draw, (10, 10), f"Frame: {frame_num}", font, (255, 255, 255), (0, 0, 0))
        
        # Add effect annotations with larger, bolder text
        effect1_color = (0, 255, 0) if effect1 == 1 else (255, 0, 0)
        effect2_color = (0, 255, 0) if effect2 == 1 else (255, 0, 0)
        
        self.draw_text_with_outline(draw, (10, 70), f"{self.dataset_type}_set.json: Effect {effect1}", 
                                   font, effect1_color, (0, 0, 0))
        self.draw_text_with_outline(draw, (10, 130), f"{self.dataset_type}_set_peat.json: Effect {effect2}", 
                                   font, effect2_color, (0, 0, 0))
        
        # Add difference indicator with larger text
        self.draw_text_with_outline(draw, (10, 190), "DIFFERENCE DETECTED!", 
                                   font, (255, 255, 0), (255, 0, 0))
        
        return pil_image
    
    def create_gif_for_range(self, start_frame: int, end_frame: int, 
                           annotations1: np.ndarray, annotations2: np.ndarray,
                           output_path: str):
        """
        Create a GIF for a specific frame range where differences occur
        Include ALL frames and calculate duration to match ~30fps playback
        
        Args:
            start_frame: Starting frame
            end_frame: Ending frame  
            annotations1: Annotations from first JSON
            annotations2: Annotations from second JSON
            output_path: Path to save GIF
        """
        frames_data = self.extract_frames_from_range(start_frame, end_frame)
        gif_frames = []
        
        # Calculate duration to achieve ~30fps playback
        # 1000ms / 30fps = ~33ms per frame
        frame_duration = 33
        
        print(f"Creating GIF with {len(frames_data)} frames at ~30fps ({frame_duration}ms per frame)")
        
        for frame_num, frame in frames_data:
            effect1 = annotations1[frame_num]
            effect2 = annotations2[frame_num]
            
            comparison_image = self.create_comparison_image(frame_num, frame, effect1, effect2)
            gif_frames.append(comparison_image)
        
        if gif_frames:
            gif_frames[0].save(
                output_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=frame_duration,  # ~30fps
                loop=0
            )
            print(f"Created GIF: {output_path} ({len(gif_frames)} frames)")
        else:
            print(f"Warning: No frames extracted for range {start_frame}-{end_frame}")
    
    def create_visualization_report(self, annotations1: np.ndarray, annotations2: np.ndarray, 
                                  difference_mask: np.ndarray):
        """
        Create beautiful visualization report showing the differences
        """
        # Set style for beautiful plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 0.8, 0.6], hspace=0.3, wspace=0.1)
        
        # Create time axis
        time_axis = np.arange(len(annotations1)) / self.fps
        
        # Define colors
        colors = {
            'effect_0': '#FF6B6B',  # Red for effect 0
            'effect_1': '#4ECDC4',  # Teal for effect 1
            'unannotated': '#E8E8E8',  # Light gray for unannotated
            'difference': '#FFD93D',  # Yellow for differences
            'json1': '#6C5CE7',  # Purple
            'json2': '#00B894'   # Green
        }
        
        # Plot 1: val_set.json annotations with better styling
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_annotations(ax1, time_axis, annotations1, 
                             f'{self.dataset_type}_set.json Annotations', 
                             colors['json1'])
        
        # Plot 2: val_set_peat.json annotations
        ax2 = fig.add_subplot(gs[1, :])
        self.plot_annotations(ax2, time_axis, annotations2, 
                             f'{self.dataset_type}_set_peat.json Annotations', 
                             colors['json2'])
        
        # Plot 3: Side-by-side comparison
        ax3 = fig.add_subplot(gs[2, :])
        self.plot_comparison(ax3, time_axis, annotations1, annotations2, colors)
        
        # Plot 4: Difference analysis
        ax4 = fig.add_subplot(gs[3, :])
        self.plot_differences(ax4, time_axis, difference_mask, colors)
        
        # Plot 5: Statistics summary
        ax5 = fig.add_subplot(gs[4, :])
        self.plot_statistics(ax5, annotations1, annotations2, difference_mask, colors)
        
        plt.suptitle(f'Video Annotation Analysis: {self.dataset_type.upper()} Dataset', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(f'analytic/{self.dataset_type}/annotation_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
    def plot_annotations(self, ax, time_axis, annotations, title, color):
        """Plot annotations with better styling"""
        # Get annotation coverage from original JSON data
        annotated_mask = self.get_annotated_mask(title)
        
        if np.any(annotated_mask):
            # Plot effect 1 regions
            effect1_mask = annotated_mask & (annotations == 1)
            if np.any(effect1_mask):
                ax.fill_between(time_axis, 0, 1, where=effect1_mask, 
                               color='#4ECDC4', alpha=0.8, label='Effect 1', step='pre')
            
            # Plot effect 0 regions  
            effect0_mask = annotated_mask & (annotations == 0)
            if np.any(effect0_mask):
                ax.fill_between(time_axis, 0, 1, where=effect0_mask, 
                               color='#FF6B6B', alpha=0.8, label='Effect 0', step='pre')
        
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Effect', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No Effect', 'Effect'])
        
    def get_annotated_mask(self, title):
        """Get mask of which frames are actually annotated in the original JSON"""
        annotated_mask = np.zeros(self.total_frames, dtype=bool)
        
        # Determine which dataset to use based on title
        if 'peat' in title.lower():
            data = self.data2
        else:
            data = self.data1
            
        for video_data in data:
            for segment in video_data['segments']:
                start_frame = segment['start_frame']
                end_frame = min(segment['end_frame'], self.total_frames)
                annotated_mask[start_frame:end_frame] = True
                
        return annotated_mask
        
    def plot_comparison(self, ax, time_axis, annotations1, annotations2, colors):
        """Plot overlaid comparison"""
        annotated1 = self.get_annotated_mask('original')
        annotated2 = self.get_annotated_mask('peat')
        
        # Plot first annotation set
        if np.any(annotated1):
            effect1_1 = annotated1 & (annotations1 == 1)
            effect0_1 = annotated1 & (annotations1 == 0)
            if np.any(effect1_1):
                ax.fill_between(time_axis, 0.5, 1, where=effect1_1, 
                               color=colors['json1'], alpha=0.6, 
                               label=f'{self.dataset_type}_set.json', step='pre')
            if np.any(effect0_1):
                ax.fill_between(time_axis, 0.5, 1, where=effect0_1, 
                               color=colors['json1'], alpha=0.3, step='pre')
        
        # Plot second annotation set
        if np.any(annotated2):
            effect1_2 = annotated2 & (annotations2 == 1)
            effect0_2 = annotated2 & (annotations2 == 0)
            if np.any(effect1_2):
                ax.fill_between(time_axis, 0, 0.5, where=effect1_2, 
                               color=colors['json2'], alpha=0.6, 
                               label=f'{self.dataset_type}_set_peat.json', step='pre')
            if np.any(effect0_2):
                ax.fill_between(time_axis, 0, 0.5, where=effect0_2, 
                               color=colors['json2'], alpha=0.3, step='pre')
        
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Annotations', fontsize=12, fontweight='bold')
        ax.set_title('Comparison (Top: Original, Bottom: Peat)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
    def plot_differences(self, ax, time_axis, difference_mask, colors):
        """Plot difference points"""
        if np.any(difference_mask):
            diff_times = time_axis[difference_mask]
            ax.scatter(diff_times, np.ones(len(diff_times)), 
                      c=colors['difference'], s=30, alpha=0.8, 
                      label=f'Differences ({len(diff_times)} frames)')
            ax.fill_between(time_axis, 0, 1, where=difference_mask, 
                           color=colors['difference'], alpha=0.3, step='pre')
        
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Difference', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Difference Points', fontsize=14, fontweight='bold', color=colors['difference'])
        ax.grid(True, alpha=0.3, linestyle='--')
        if np.any(difference_mask):
            ax.legend(loc='upper right', framealpha=0.9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Same', 'Different'])
        
    def plot_statistics(self, ax, annotations1, annotations2, difference_mask, colors):
        """Plot statistics summary"""
        ax.axis('off')
        
        # Get actual annotated frames
        annotated1 = self.get_annotated_mask('original')
        annotated2 = self.get_annotated_mask('peat')
        total_annotated = np.sum(annotated1 & annotated2)
        total_differences = np.sum(difference_mask)
        
        if total_annotated > 0:
            diff_percentage = (total_differences / total_annotated) * 100
        else:
            diff_percentage = 0
            
        effect1_count1 = np.sum(annotated1 & (annotations1 == 1))
        effect0_count1 = np.sum(annotated1 & (annotations1 == 0))
        effect1_count2 = np.sum(annotated2 & (annotations2 == 1))
        effect0_count2 = np.sum(annotated2 & (annotations2 == 0))
        
        # Create statistics text
        stats_text = f"""
STATISTICS SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Total Frames: {self.total_frames:,}
📋 Annotated Frames: {total_annotated:,}
⚠️  Difference Frames: {total_differences:,} ({diff_percentage:.2f}%)

📁 {self.dataset_type}_set.json:        Effect 0: {effect0_count1:,} | Effect 1: {effect1_count1:,}
📁 {self.dataset_type}_set_peat.json:   Effect 0: {effect0_count2:,} | Effect 1: {effect1_count2:,}
        """
        
        ax.text(0.05, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
    def generate_report(self):
        """
        Generate comprehensive diagnostic report
        """
        print("=" * 60)
        print("VIDEO ANNOTATION DIAGNOSTIC REPORT")
        print("=" * 60)
        
        # Find differences
        difference_mask, difference_ranges, annotations1, annotations2 = self.find_differences()
        
        # Get actual annotated frames for statistics
        annotated1 = self.get_annotated_mask('original') 
        annotated2 = self.get_annotated_mask('peat')
        
        # Total frames to compare = all frames in original (since peat frames not in original are ignored)
        total_annotated_frames = np.sum(annotated1)
        total_diff_frames = np.sum(difference_mask)
        
        print(f"\nBASIC STATISTICS:")
        print(f"Total frames in video: {self.total_frames}")
        print(f"Total frames in original JSON: {total_annotated_frames}")
        print(f"Frames also in peat JSON: {np.sum(annotated1 & annotated2)}")
        print(f"Frames in original but NOT in peat (marked as 0): {np.sum(annotated1 & ~annotated2)}")
        print(f"Frames with differences: {total_diff_frames}")
        if total_annotated_frames > 0:
            print(f"Difference percentage: {(total_diff_frames/total_annotated_frames)*100:.2f}%")
        
        # Effect distribution - now considering the special peat logic
        print(f"\nEFFECT DISTRIBUTION:")
        effect1_count1 = np.sum(annotated1 & (annotations1 == 1))
        effect0_count1 = np.sum(annotated1 & (annotations1 == 0))
        
        # For peat: count actual annotations + frames marked as 0 (not converted)
        peat_effect1 = np.sum(annotated2 & (annotations2 == 1))
        peat_effect0_annotated = np.sum(annotated2 & (annotations2 == 0))
        peat_effect0_not_converted = np.sum(annotated1 & ~annotated2)
        peat_effect0_total = peat_effect0_annotated + peat_effect0_not_converted
        
        print(f"{self.dataset_type}_set.json - Effect 0: {effect0_count1}, Effect 1: {effect1_count1}")
        print(f"{self.dataset_type}_set_peat.json - Effect 0: {peat_effect0_total} (annotated: {peat_effect0_annotated}, not converted: {peat_effect0_not_converted}), Effect 1: {peat_effect1}")
        
        # Difference ranges
        print(f"\nDIFFERENCE RANGES:")
        print(f"Found {len(difference_ranges)} difference ranges:")
        
        # Create output directory for this dataset type
        os.makedirs(f'analytic/{self.dataset_type}', exist_ok=True)
        
        for i, (start, end) in enumerate(difference_ranges):
            duration = (end - start + 1) / self.fps
            start_time = start / self.fps
            end_time = end / self.fps
            frame_count = end - start + 1
            
            print(f"  Range {i+1}: Frames {start}-{end} "
                  f"(Time: {start_time:.2f}s-{end_time:.2f}s, Duration: {duration:.2f}s, {frame_count} frames)")
            
            # Create GIF for this range with ALL frames
            gif_path = f'analytic/{self.dataset_type}/difference_range_{i+1}_frames_{start}_{end}.gif'
            self.create_gif_for_range(start, end, annotations1, annotations2, gif_path)
        
        if len(difference_ranges) > 20:
            print(f"\nNote: Processing all {len(difference_ranges)} difference ranges.")
            print(f"This may take some time for large ranges with many frames.")
            print(f"Each GIF will include ALL frames at ~30fps playback speed.")
        
        # Create visualization
        self.create_visualization_report(annotations1, annotations2, difference_mask)
        
        # Save detailed data
        self.save_detailed_analysis(annotations1, annotations2, difference_mask, difference_ranges)
        
        print(f"\nOUTPUT FILES CREATED:")
        print(f"- analytic/{self.dataset_type}/annotation_comparison.png - Visual comparison chart")
        print(f"- analytic/{self.dataset_type}/detailed_analysis.json - Detailed difference data")
        print(f"- analytic/{self.dataset_type}/difference_range_*.gif - GIFs showing difference frames")
        
    def save_detailed_analysis(self, annotations1: np.ndarray, annotations2: np.ndarray, 
                              difference_mask: np.ndarray, difference_ranges: List[Tuple[int, int]]):
        """
        Save detailed analysis to JSON file
        """
        analysis_data = {
            "summary": {
                "total_frames": self.total_frames,
                "fps": self.fps,
                "total_differences": int(np.sum(difference_mask)),
                "difference_ranges_count": len(difference_ranges)
            },
            "difference_ranges": [
                {
                    "range_id": i + 1,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "start_time_seconds": float(start / self.fps),
                    "end_time_seconds": float(end / self.fps),
                    "duration_seconds": float((end - start + 1) / self.fps),
                    "frame_count": int(end - start + 1)
                }
                for i, (start, end) in enumerate(difference_ranges)
            ],
            "frame_by_frame_differences": [
                {
                    "frame": int(i),
                    "time_seconds": float(i / self.fps),
                    "val_set_effect": int(annotations1[i]) if annotations1[i] != -1 else None,
                    "val_set_peat_effect": int(annotations2[i]) if annotations2[i] != -1 else None,
                    "differs": bool(difference_mask[i])
                }
                for i in range(len(difference_mask)) if difference_mask[i]
            ]
        }
        
        with open(f'analytic/{self.dataset_type}/detailed_analysis.json', 'w') as f:
            json.dump(analysis_data, f, indent=2)

# Usage example
if __name__ == "__main__":
    import argparse
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Video Annotation Diagnostic Tool')
    parser.add_argument('--dataset', choices=['val', 'test', 'train'], default='val',
                       help='Dataset type to analyze (val, test, or train)')
    
    args = parser.parse_args()
    
    # Initialize the diagnostic tool with specified dataset type
    diagnostic = VideoAnnotationDiagnostic(dataset_type=args.dataset)
    
    # Generate comprehensive report
    diagnostic.generate_report()
    
    print(f"\nDiagnostic complete for {args.dataset} dataset! Check the 'analytic/{args.dataset}' folder for output files.")
    
    # Example usage without command line args:
    # diagnostic = VideoAnnotationDiagnostic('val')    # for val dataset
    # diagnostic = VideoAnnotationDiagnostic('test')   # for test dataset  
    # diagnostic = VideoAnnotationDiagnostic('train')  # for train dataset