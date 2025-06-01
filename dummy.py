import cv2
import mediapipe as mp
import numpy as np
import json
import requests
from io import BytesIO
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import yt_dlp
from urllib.parse import urlparse

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
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True
        )
        
        # YouTube downloader configuration
        self.ydl_opts = {
            'format': 'best[height<=720]',  # Limit quality to reduce processing time
            'quiet': True,
            'no_warnings': True,
        }
    
    def extract_keypoints_from_url(self, url: str, start_time: float = 0.0, 
                                 end_time: Optional[float] = None, 
                                 max_frames: int = 100) -> Dict:
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
        
        # Determine if it's a YouTube URL
        if 'youtube.com' in url or 'youtu.be' in url:
            return self._extract_from_youtube(url, start_time, end_time, max_frames)
        else:
            return self._extract_from_direct_url(url, start_time, end_time, max_frames)
    
    def _extract_from_youtube(self, url: str, start_time: float, 
                            end_time: Optional[float], max_frames: int) -> Dict:
        """Extract keypoints from YouTube video"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Get video stream URL
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_url = info['url']
                
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
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        if end_time:
            end_frame = int(end_time * fps)
            max_frames = min(max_frames, end_frame - start_frame)
        else:
            end_frame = min(total_frames, start_frame + max_frames)
        
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
                if current_frame_num >= end_frame:
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
                    print(f"Processed {frame_count}/{max_frames} frames")
        
        finally:
            cap.release()
        
        keypoints_data["video_info"]["processed_frames"] = processed_frames
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
        
        # Extract face keypoints
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
        return [[lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0] 
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
            print(f"\nProcessing sample {i+1}/{max_samples}: {entry['text']}")
            
            # Extract keypoints
            keypoints = self.extract_keypoints_from_url(
                url=entry['url'],
                start_time=entry.get('start_time', 0.0),
                end_time=entry.get('end_time'),
                max_frames=60  # Limit frames for demo
            )
            
            # Combine original data with keypoints
            result = {
                "original_data": entry,
                "keypoints_data": keypoints
            }
            
            results.append(result)
            
            # Save progress
            if (i + 1) % 5 == 0:
                self._save_results(results, f"{output_file}_progress.json")
        
        # Save final results
        self._save_results(results, output_file)
        print(f"\nSaved keypoints for {len(results)} samples to {output_file}")
    
    def _save_results(self, results: List[Dict], filename: str) -> None:
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    def visualize_keypoints(self, frame: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        Visualize keypoints on frame
        
        Args:
            frame: Input frame
            keypoints: Keypoints dictionary
            
        Returns:
            Frame with keypoints drawn
        """
        
        annotated_frame = frame.copy()
        
        # Convert keypoints back to MediaPipe format for visualization
        # This is a simplified version - you might need to adjust based on your needs
        
        return annotated_frame

# Example usage
def main():
    # Initialize extractor
    extractor = ASLKeypointExtractor()
    
    # Example MS-ASL data (from your sample)
    sample_data = [
        {
            "org_text": "match [light-a-MATCH]",
            "clean_text": "match",
            "start_time": 0.0,
            "end_time": 2.767,
            "url": "https://www.youtube.com/watch?v=C37R_Ix8-qs",
            "text": "match"
        },
        {
            "org_text": "FAIL",
            "clean_text": "fail",
            "start_time": 0.0,
            "end_time": 2.96,
            "url": "https://www.youtube.com/watch?v=PIsUJl8BN_I",
            "text": "fail"
        }
    ]
    
    # Process sample data
    extractor.process_msasl_dataset(
        json_data=sample_data,
        output_file="msasl_keypoints.json",
        max_samples=2
    )
    
    # Or extract keypoints from a single video
    single_result = extractor.extract_keypoints_from_url(
        url="https://www.youtube.com/watch?v=C37R_Ix8-qs",
        start_time=0.0,
        end_time=2.767,
        max_frames=50
    )
    
    print("Single video keypoints extracted:", len(single_result.get("keypoints", [])))

if __name__ == "__main__":
    main()