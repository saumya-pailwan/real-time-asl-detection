#!/usr/bin/env python3
"""
Complete ASL Keypoint Extractor - Single File Version
Run this directly to test and process your MS-ASL dataset
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import requests
from io import BytesIO
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import argparse

# Try to import yt-dlp, fall back to youtube-dl if needed
try:
    import yt_dlp
    USE_YT_DLP = True
except ImportError:
    try:
        import youtube_dl as yt_dlp
        USE_YT_DLP = False
        print("Warning: Using youtube-dl instead of yt-dlp. Consider upgrading: pip install yt-dlp")
    except ImportError:
        print("Error: Neither yt-dlp nor youtube-dl found. Install with: pip install yt-dlp")
        exit(1)

class ASLKeypointExtractor:
    """
    Extract keypoints from ASL videos on the fly without downloading them permanently.
    Supports YouTube URLs and direct video URLs.
    """
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # Reduced for better performance
            enable_segmentation=False,
            refine_face_landmarks=False  # Disabled for better performance
        )
        
        # YouTube downloader configuration
        if USE_YT_DLP:
            self.ydl_opts = {
                'format': 'best[height<=480]',  # Lower quality for faster processing
                'quiet': True,
                'no_warnings': True,
            }
        else:
            self.ydl_opts = {
                'format': 'best[height<=480]',
                'quiet': True,
            }
    
    def extract_keypoints_from_url(self, url: str, start_time: float = 0.0, 
                                 end_time: Optional[float] = None, 
                                 max_frames: int = 50) -> Dict:
        """
        Extract keypoints from video URL (YouTube or direct video URL)
        
        Args:
            url: Video URL
            start_time: Start time in seconds
            end_time: End time in seconds (if None, uses max_frames)
            max_frames: Maximum number of frames to process
            
        Returns:
            Dict containing keypoints data
        """
        
        print(f"Processing: {url}")
        
        # Determine if it's a YouTube URL
        if 'youtube.com' in url or 'youtu.be' in url:
            return self._extract_from_youtube(url, start_time, end_time, max_frames)
        else:
            return self._extract_from_direct_url(url, start_time, end_time, max_frames)
    
    def _extract_from_youtube(self, url: str, start_time: float, 
                            end_time: Optional[float], max_frames: int) -> Dict:
        """Extract keypoints from YouTube video"""
        
        try:
            # Get video stream URL
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if USE_YT_DLP:
                    video_url = info['url']
                else:
                    # For youtube-dl
                    formats = info.get('formats', [])
                    if formats:
                        video_url = formats[-1]['url']  # Get best available format
                    else:
                        return {"error": "No video formats found"}
            
            # Process video stream
            return self._process_video_stream(video_url, start_time, end_time, max_frames)
            
        except Exception as e:
            print(f"Error processing YouTube video: {e}")
            return {"error": str(e)}
    
    def _extract_from_direct_url(self, url: str, start_time: float, 
                               end_time: Optional[float], max_frames: int) -> Dict:
        """Extract keypoints from direct video URL"""
        return self._process_video_stream(url, start_time, end_time, max_frames)
    
    def _process_video_stream(self, video_url: str, start_time: float, 
                            end_time: Optional[float], max_frames: int) -> Dict:
        """Process video stream and extract keypoints"""
        
        cap = cv2.VideoCapture(video_url)
        
        if not cap.isOpened():
            return {"error": "Could not open video stream"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 30  # Default FPS if unable to detect
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        if end_time:
            end_frame = int(end_time * fps)
            max_frames = min(max_frames, end_frame - start_frame)
        else:
            end_frame = min(total_frames, start_frame + max_frames) if total_frames > 0 else start_frame + max_frames
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        keypoints_data = {
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": 0,
                "start_time": start_time,
                "end_time": end_time
            },
            "keypoints": []
        }
        
        frame_count = 0
        processed_frames = 0
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame_num = start_frame + frame_count
                if end_time and current_frame_num >= end_frame:
                    break
                
                # Process frame
                keypoints = self._extract_frame_keypoints(frame)
                
                if keypoints:
                    keypoints_data["keypoints"].append({
                        "frame_number": current_frame_num,
                        "timestamp": current_frame_num / fps,
                        "pose": keypoints.get("pose"),
                        "face": keypoints.get("face"),
                        "left_hand": keypoints.get("left_hand"),
                        "right_hand": keypoints.get("right_hand")
                    })
                    processed_frames += 1
                
                frame_count += 1
                
                # Progress indicator
                if frame_count % 10 == 0:
                    print(f"  Processed {frame_count}/{max_frames} frames")
        
        finally:
            cap.release()
        
        keypoints_data["video_info"]["processed_frames"] = processed_frames
        print(f"  Completed: {processed_frames} frames processed")
        return keypoints_data
    
    def _extract_frame_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract keypoints from a single frame"""
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.holistic.process(rgb_frame)
        
        keypoints = {}
        
        # Extract pose keypoints
        if results.pose_landmarks:
            keypoints["pose"] = self._landmarks_to_array(results.pose_landmarks)
        
        # Extract face keypoints (if enabled)
        if results.face_landmarks:
            keypoints["face"] = self._landmarks_to_array(results.face_landmarks)
        
        # Extract hand keypoints
        if results.left_hand_landmarks:
            keypoints["left_hand"] = self._landmarks_to_array(results.left_hand_landmarks)
        
        if results.right_hand_landmarks:
            keypoints["right_hand"] = self._landmarks_to_array(results.right_hand_landmarks)
        
        return keypoints if keypoints else None
    
    def _landmarks_to_array(self, landmarks) -> List[List[float]]:
        """Convert MediaPipe landmarks to array format"""
        return [[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)] 
                for lm in landmarks.landmark]
    
    def process_msasl_dataset(self, json_data: List[Dict], output_file: str, 
                            max_samples: int = 10) -> None:
        """
        Process MS-ASL dataset entries and extract keypoints
        
        Args:
            json_data: List of MS-ASL dataset entries
            output_file: Output file path for keypoints
            max_samples: Maximum number of samples to process
        """
        
        results = []
        
        for i, entry in enumerate(json_data[:max_samples]):
            print(f"\nProcessing sample {i+1}/{max_samples}: {entry.get('text', 'Unknown')}")
            
            # Extract keypoints
            keypoints = self.extract_keypoints_from_url(
                url=entry['url'],
                start_time=entry.get('start_time', 0.0),
                end_time=entry.get('end_time'),
                max_frames=30  # Reduced for faster processing
            )
            
            # Combine original data with keypoints
            result = {
                "original_data": entry,
                "keypoints_data": keypoints
            }
            
            results.append(result)
            
            # Save progress every 5 samples
            if (i + 1) % 5 == 0:
                progress_file = f"{output_file.rsplit('.', 1)[0]}_progress_{i+1}.json"
                self._save_results(results, progress_file)
                print(f"  Progress saved to {progress_file}")
        
        # Save final results
        self._save_results(results, output_file)
        print(f"\n Saved keypoints for {len(results)} samples to {output_file}")
    
    def _save_results(self, results: List[Dict], filename: str) -> None:
        """Save results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving results: {e}")

def test_dependencies():
    """Test if all dependencies are working"""
    print("Testing dependencies...")
    
    try:
        import cv2
        print(" OpenCV - OK")
    except ImportError:
        print(" OpenCV - MISSING (pip install opencv-python)")
        return False
    
    try:
        import mediapipe
        print(" MediaPipe - OK")
    except ImportError:
        print(" MediaPipe - MISSING (pip install mediapipe)")
        return False
    
    try:
        import numpy
        print(" NumPy - OK")
    except ImportError:
        print(" NumPy - MISSING (pip install numpy)")
        return False
    
    if USE_YT_DLP:
        print(" yt-dlp - OK")
    else:
        print("  Using youtube-dl (consider upgrading to yt-dlp)")
    
    return True

def run_quick_test():
    """Run a quick test with sample data"""
    print("\n" + "="*50)
    print("Running Quick Test")
    print("="*50)
    
    if not test_dependencies():
        return False
    
    # Sample data from your dataset
    sample_data = [
        {
            "org_text": "match [light-a-MATCH]",
            "clean_text": "match",
            "start_time": 0.0,
            "end_time": 2.0,  # Shortened for quick test
            "url": "https://www.youtube.com/watch?v=C37R_Ix8-qs",
            "text": "match"
        }
    ]
    
    # Initialize extractor
    extractor = ASLKeypointExtractor()
    
    try:
        # Test single video
        print("\nTesting single video extraction...")
        result = extractor.extract_keypoints_from_url(
            url=sample_data[0]['url'],
            start_time=0.0,
            end_time=2.0,
            max_frames=20
        )
        
        if "error" in result:
            print(f" Error: {result['error']}")
            return False
        
        print(f" Success! Processed {result['video_info']['processed_frames']} frames")
        
        # Save test result
        with open('quick_test_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(" Test result saved to 'quick_test_result.json'")
        
        return True
        
    except Exception as e:
        print(f" Test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='ASL Keypoint Extractor')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--input', '-i', help='Input MS-ASL JSON file')
    parser.add_argument('--output', '-o', help='Output keypoints JSON file')
    parser.add_argument('--max-samples', '-n', type=int, default=10, help='Maximum samples to process')
    
    args = parser.parse_args()
    
    if args.test:
        success = run_quick_test()
        if success:
            print("\n Test passed! You can now process your dataset.")
        else:
            print("\n Test failed. Please fix the issues above.")
        return
    
    if args.input and args.output:
        print("="*60)
        print("Processing MS-ASL Dataset")
        print("="*60)
        
        # Load dataset
        try:
            with open(args.input, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} samples from {args.input}")
        except Exception as e:
            print(f" Error loading dataset: {e}")
            return
        
        # Process dataset
        extractor = ASLKeypointExtractor()
        extractor.process_msasl_dataset(
            json_data=data,
            output_file=args.output,
            max_samples=args.max_samples
        )
    else:
        print("ASL Keypoint Extractor")
        print("\nUsage:")
        print("  python asl_extractor_complete.py --test                    # Run quick test")
        print("  python asl_extractor_complete.py -i input.json -o output.json -n 10")
        print("\nFirst run --test to make sure everything works!")

if __name__ == "__main__":
    main()