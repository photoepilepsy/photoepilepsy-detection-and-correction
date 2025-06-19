# -*- coding: utf-8 -*-
"""
Video Effect Detection Inference Script
Processes a video file and generates effect predictions with timestamps
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import time

# Model parameters (must match training configuration)
MAX_FRAMES = 30
FRAME_SIZE = (224, 224)
FEATURE_DIM = 1024
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations for video frames (same as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FrameFeatureExtractor(nn.Module):
    """Feature extractor using EfficientNet-B0"""
    def __init__(self, pretrained=True):
        super(FrameFeatureExtractor, self).__init__()
        base_model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, FEATURE_DIM)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TransformerClassifier(nn.Module):
    """Transformer-based video effect classifier"""
    def __init__(self):
        super(TransformerClassifier, self).__init__()
        self.feature_extractor = FrameFeatureExtractor()
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_FRAMES, FEATURE_DIM) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FEATURE_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FEATURE_DIM * 4,
            dropout=DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape for feature extraction
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Reshape back
        features = features.view(batch_size, num_frames, -1)
        
        # Add positional encoding
        features = features + self.pos_encoder
        
        # Apply transformer
        transformer_output = self.transformer_encoder(features)
        
        # Global pooling
        pooled_output = torch.mean(transformer_output, dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits.squeeze(-1)

class VideoInference:
    """Video inference class for effect detection"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to the saved model weights
            confidence_threshold: Threshold for binary classification
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.transform = transform
        
        print(f"🔧 Initializing inference engine...")
        print(f"📱 Device: {DEVICE}")
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            print(f"📂 Loading model from: {self.model_path}")
            
            # Initialize model
            self.model = TransformerClassifier().to(DEVICE)
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"✅ Model loaded successfully!")
            print(f"🔢 Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _extract_frames_window(self, video_path: str, start_frame: int, num_frames: int) -> torch.Tensor:
        """
        Extract a window of frames from video
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            num_frames: Number of frames to extract
            
        Returns:
            Tensor of processed frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        
        for i in range(num_frames):
            frame_idx = start_frame + i
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # If we can't read more frames, pad with the last frame
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    # Create a black frame if no frames were read
                    black_frame = torch.zeros(3, FRAME_SIZE[0], FRAME_SIZE[1])
                    frames.append(black_frame)
                continue
            
            # Resize and convert color
            frame = cv2.resize(frame, FRAME_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)
        
        cap.release()
        
        # Ensure we have exactly num_frames
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1].clone())
            else:
                black_frame = torch.zeros(3, FRAME_SIZE[0], FRAME_SIZE[1])
                frames.append(black_frame)
        
        return torch.stack(frames)
    
    def predict_window(self, frames_tensor: torch.Tensor) -> Tuple[int, float]:
        """
        Predict effect for a window of frames
        
        Args:
            frames_tensor: Tensor of frames [num_frames, channels, height, width]
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Add batch dimension
        frames_batch = frames_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(frames_batch)
            probability = torch.sigmoid(logits).item()
            prediction = 1 if probability > self.confidence_threshold else 0
        
        return prediction, probability
    
    def load_ground_truth(self, json_path: str) -> Dict[int, int]:
        """
        Load ground truth labels from JSON file
        
        Args:
            json_path: Path to ground truth JSON file
            
        Returns:
            Dictionary mapping frame indices to effect labels
        """
        print(f"📋 Loading ground truth from: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        effect_map = {}
        
        # Handle both single video data and list of videos
        if isinstance(data, list):
            video_data_list = data
        else:
            video_data_list = [data]
        
        # First pass: Initialize all frames to 0 (no effect)
        total_frames_marked = 0
        
        for video_data in video_data_list:
            if "segments" in video_data:
                for segment in video_data["segments"]:
                    start_frame = segment["start_frame"]
                    end_frame = segment["end_frame"]
                    effect = segment["effect"]
                    
                    # Mark every frame in this segment
                    for frame_idx in range(start_frame, end_frame):
                        # Key logic: If frame is already marked as 1 (effect), don't change it
                        # Only update if frame is not yet marked or current effect is 1
                        if frame_idx not in effect_map:
                            effect_map[frame_idx] = effect
                        elif effect == 1:
                            # If current segment has effect=1, override any previous 0
                            effect_map[frame_idx] = 1
                        # If effect=0 and frame already has 1, keep it as 1 (don't downgrade)
                    
                    total_frames_marked += (end_frame - start_frame)
        
        # Count positive and negative frames
        positive_frames = sum(1 for label in effect_map.values() if label == 1)
        negative_frames = len(effect_map) - positive_frames
        
        print(f"✅ Loaded ground truth for {len(effect_map)} frames")
        
        return effect_map
    
    def calculate_window_ground_truth(self, effect_map: Dict[int, int], start_frame: int, num_frames: int) -> int:
        """
        Calculate ground truth label for a window based on majority vote
        Uses same logic as training: effect=1 if more than 1/3 of frames have effect
        Key principle: Once a frame is marked as positive (effect=1), it stays positive
        
        Args:
            effect_map: Dictionary mapping frame indices to labels
            start_frame: Start frame of window
            num_frames: Number of frames in window (typically 30 for 30fps, 1-second window)
            
        Returns:
            Ground truth label (0 or 1)
        """
        frame_labels = []
        positive_frames = 0
        
        for i in range(start_frame, start_frame + num_frames):
            # Get label for this frame, default to 0 if not in map
            label = effect_map.get(i, 0)
            frame_labels.append(label)
            if label == 1:
                positive_frames += 1
        
        # Use same logic as training: effect if more than 1/3 of frames have effect
        # This matches the training dataset generation logic
        threshold = len(frame_labels) / 3
        window_label = 1 if positive_frames > threshold else 0
        
        return window_label
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """
        Calculate sequence-level metrics: accuracy, precision, recall, F1
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing all metrics
        """
        # Calculate basic counts
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
        
        total = len(y_true)
        correct = tp + tn
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_sequences": total,
            "correct_predictions": correct,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        }
        
        return metrics
    
    def process_video(self, video_path: str, window_stride: int = 30, output_path: str = None, 
                     ground_truth_path: str = None) -> Dict:
        """
        Process entire video and generate predictions
        
        Args:
            video_path: Path to input video
            window_stride: Stride between windows (frames)
            output_path: Path to save results JSON
            ground_truth_path: Path to ground truth JSON file (optional)
            
        Returns:
            Dictionary containing predictions and metadata
        """
        print(f"🎬 Processing video: {video_path}")
        
        # Load ground truth if provided
        ground_truth_map = None
        if ground_truth_path:
            ground_truth_map = self.load_ground_truth(ground_truth_path)
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()
        
        # Calculate number of windows
        num_windows = max(1, (total_frames - MAX_FRAMES) // window_stride + 1)
        
        # Process windows
        results = []
        effect_segments = []
        current_segment = None
        
        # For evaluation metrics
        y_true = []
        y_pred = []
        
        print(f"\n🔄 Processing windows...")
        start_time = time.time()
        
        for window_idx in tqdm(range(num_windows), desc="Processing"):
            start_frame = window_idx * window_stride
            end_frame = min(start_frame + MAX_FRAMES, total_frames)
            
            # Extract frames
            try:
                frames_tensor = self._extract_frames_window(video_path, start_frame, MAX_FRAMES)
                prediction, confidence = self.predict_window(frames_tensor)
                
                # Convert frame indices to timestamps
                start_time_sec = start_frame / fps
                end_time_sec = end_frame / fps
                
                # Calculate ground truth for this window if available
                ground_truth = None
                if ground_truth_map is not None:
                    ground_truth = self.calculate_window_ground_truth(ground_truth_map, start_frame, MAX_FRAMES)
                    y_true.append(ground_truth)
                    y_pred.append(prediction)
                
                window_result = {
                    "window_id": window_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time_sec,
                    "end_time": end_time_sec,
                    "prediction": prediction,
                    "confidence": confidence
                }
                
                # Add ground truth to result if available
                if ground_truth is not None:
                    window_result["ground_truth"] = ground_truth
                    window_result["correct"] = prediction == ground_truth
                
                results.append(window_result)
                
                # Track effect segments
                if prediction == 1:  # Effect detected
                    if current_segment is None:
                        # Start new segment
                        current_segment = {
                            "start_frame": start_frame,
                            "start_time": start_time_sec,
                            "end_frame": end_frame,
                            "end_time": end_time_sec,
                            "max_confidence": confidence
                        }
                    else:
                        # Extend current segment
                        current_segment["end_frame"] = end_frame
                        current_segment["end_time"] = end_time_sec
                        current_segment["max_confidence"] = max(current_segment["max_confidence"], confidence)
                else:  # No effect
                    if current_segment is not None:
                        # End current segment
                        effect_segments.append(current_segment)
                        current_segment = None
                
                # Log progress periodically
                if (window_idx + 1) % 100 == 0:
                    print(f"   Processed {window_idx + 1}/{num_windows} windows")
                
            except Exception as e:
                print(f"⚠️  Error processing window {window_idx}: {e}")
                # Add error result
                window_result = {
                    "window_id": window_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_frame / fps,
                    "end_time": end_frame / fps,
                    "prediction": 0,
                    "confidence": 0.0,
                    "error": str(e)
                }
                
                # Add ground truth if available
                if ground_truth_map is not None:
                    ground_truth = self.calculate_window_ground_truth(ground_truth_map, start_frame, MAX_FRAMES)
                    window_result["ground_truth"] = ground_truth
                    window_result["correct"] = False  # Error means incorrect
                    y_true.append(ground_truth)
                    y_pred.append(0)  # Default prediction for errors
                
                results.append(window_result)
        
        # Close any remaining segment
        if current_segment is not None:
            effect_segments.append(current_segment)
        
        # Calculate statistics
        total_effect_windows = sum(1 for r in results if r["prediction"] == 1)
        processing_time = time.time() - start_time
        
        # Calculate evaluation metrics if ground truth available
        evaluation_metrics = None
        if ground_truth_map is not None and y_true:
            evaluation_metrics = self.calculate_metrics(y_true, y_pred)
            print(f"🎯 Accuracy: {evaluation_metrics['accuracy']:.4f}")
            print(f"🔍 Precision: {evaluation_metrics['precision']:.4f}")
            print(f"📡 Recall: {evaluation_metrics['recall']:.4f}")
            print(f"🏆 F1-Score: {evaluation_metrics['f1_score']:.4f}")
        
        print(f"📈 Processing completed in {processing_time:.1f}s")
        
        # Create results dictionary
        results_dict = {
            "video_info": {
                "filename": os.path.basename(video_path),
                "total_frames": total_frames,
                "fps": fps,
                "duration_seconds": duration,
                "processing_time_seconds": processing_time
            },
            "model_info": {
                "model_path": self.model_path,
                "confidence_threshold": self.confidence_threshold,
                "window_size": MAX_FRAMES,
                "window_stride": window_stride
            },
            "statistics": {
                "total_windows": len(results),
                "effect_segments_count": len(effect_segments)
            },
            "predictions": results,
            "effect_segments": effect_segments
        }
        
        # Add evaluation metrics if available
        if evaluation_metrics is not None:
            results_dict["evaluation_metrics"] = evaluation_metrics
            results_dict["ground_truth_provided"] = True
        else:
            results_dict["ground_truth_provided"] = False
        
        # Save results if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"💾 Results saved to: {output_path}")
        
        return results_dict

def main():
    parser = argparse.ArgumentParser(description="Video Effect Detection Inference")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument("--model", "-m", required=True, help="Path to model weights (.pth file)")
    parser.add_argument("--output", "-o", help="Path to output JSON file (optional)")
    parser.add_argument("--ground_truth", "-g", help="Path to ground truth JSON file for evaluation (optional)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, 
                       help="Confidence threshold for classification (default: 0.5)")
    parser.add_argument("--stride", "-s", type=int, default=30,
                       help="Window stride in frames (default: 30)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if args.ground_truth and not os.path.exists(args.ground_truth):
        print(f"❌ Ground truth file not found: {args.ground_truth}")
        return
    
    # Set default output path if not provided
    if args.output is None:
        video_name = Path(args.video).stem
        args.output = f"{video_name}_predictions.json"
    
    print("🚀 Starting Video Effect Detection Inference")
    print("=" * 60)
    
    try:
        # Initialize inference engine
        inference = VideoInference(args.model, args.threshold)
        
        # Process video
        results = inference.process_video(
            video_path=args.video,
            window_stride=args.stride,
            output_path=args.output,
            ground_truth_path=args.ground_truth
        )
        
        print("🎉 Inference completed!")
        
        # Print evaluation metrics if available
        if results.get("ground_truth_provided", False):
            metrics = results["evaluation_metrics"]
            print(f"🎯 Final Results:")
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        
        print(f"💾 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()