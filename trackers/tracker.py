from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox

class Tracker:
    """
    A class for tracking objects (players, referees, ball) in sports video frames.
    
    The Tracker uses a pre-trained YOLO model for object detection and ByteTrack
    for object tracking across frames. It provides functionality for detecting objects,
    tracking them across frames, interpolating ball positions when detections are missed,
    and visualizing the tracking results.
    
    Attributes:
        model (YOLO): An instance of the YOLO model for object detection.
        tracker (ByteTrack): An instance of ByteTrack algorithm for object tracking.
    """
    def __init__(self, model_path):
        """
        Initialize the Tracker with a YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions across frames.
        
        When ball detection fails in some frames, this method uses pandas interpolation
        to estimate the ball's position based on adjacent frames where it was detected.
        
        Args:
            ball_positions (list): List of dictionaries containing ball bounding box data.
                Each dictionary has structure {track_id: {"bbox": [x1, y1, x2, y2]}}.
        
        Returns:
            list: List of dictionaries with interpolated ball positions.
        """
        # Extract bounding boxes, ignoring entries where ball wasn't detected
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_pos = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Fill missing values using interpolation and backward fill
        df_ball_pos = df_ball_pos.interpolate()
        df_ball_pos = df_ball_pos.bfill()

        # Convert back to the original format
        ball_positions = [{1: {"bbox": x}} for x in df_ball_pos.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        """
        Detect objects in a batch of frames using the YOLO model.
        
        Processes frames in batches to improve efficiency while maintaining memory usage.
        
        Args:
            frames (list): List of frames (as numpy arrays) to process.
            
        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = 20  # Process 20 frames at a time to balance speed and memory usage
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect and track objects across video frames.
        
        This method detects players, referees, and the ball in each frame and tracks them
        across frames. It can optionally load previously saved tracking results to avoid
        redundant computation.
        
        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool, optional): Whether to load existing tracking data. Defaults to False.
            stub_path (str, optional): Path to saved tracking data. Defaults to None.
            
        Returns:
            dict: A dictionary containing tracking data with the following structure:
                {
                    "players": [frame1_players, frame2_players, ...],
                    "referees": [frame1_referees, frame2_referees, ...],
                    "ball": [frame1_ball, frame2_ball, ...]
                }
                where each frame_X is a dictionary mapping track_ids to object data.
        """
        # Load existing tracking data if available and requested
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        # Run object detection on all frames
        detections = self.detect_frames(frames)

        # Initialize tracking dictionary
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        # Process each frame's detections
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}  # Invert map for lookup by class name
            print(cls_names)
            
            # Convert YOLO output to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Reclassify goalkeeper as player for unified tracking
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_idx] = cls_names_inv["player"]

            # Update tracker with current frame detections
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # Initialize empty dictionaries for current frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Process tracked objects (players and referees)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Process ball detections (not tracked across frames)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]      

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        
        # Save tracking results if stub_path is provided
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
        
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an elliptical marker for a player or referee at the bottom of their bounding box.
        
        Optionally displays the track ID in a rectangular label below the ellipse.
        
        Args:
            frame (numpy.ndarray): Video frame to draw on.
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].
            color (tuple): RGB color tuple for the ellipse.
            track_id (int, optional): Track ID to display. Defaults to None.
            
        Returns:
            numpy.ndarray: Frame with the ellipse drawn.
        """
        y2 = int(bbox[3])  # Bottom of the bounding box
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw elliptical marker
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Only draw ID label if track_id is provided
        if track_id is not None:
            # Define rectangle for ID text
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = int(x_center - rectangle_width//2)
            x2_rect = int(x_center + rectangle_width//2)
            y1_rect = int((y2 - rectangle_height//2) + 15)
            y2_rect = int((y2 + rectangle_height//2) + 15)

            # Draw filled rectangle for the ID label
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

            # Adjust text position for multi-digit IDs
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            # Draw track ID text
            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect+15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draw a triangular marker pointing to an object (typically the ball or player with ball).
        
        Args:
            frame (numpy.ndarray): Video frame to draw on.
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].
            color (tuple): RGB color tuple for the triangle.
            
        Returns:
            numpy.ndarray: Frame with the triangle drawn.
        """
        y = int(bbox[1])  # Top of the bounding box
        x, _ = get_center_of_bbox(bbox)

        # Define triangle points (pointing upward)
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        
        # Draw filled triangle with black outline
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw a semi-transparent overlay showing ball possession statistics.
        
        Calculates and displays the percentage of time each team has controlled the ball
        up to the current frame.
        
        Args:
            frame (numpy.ndarray): Video frame to draw on.
            frame_num (int): Current frame number.
            team_ball_control (numpy.ndarray): Array indicating which team has the ball in each frame.
                Values should be 1 for team 1, 2 for team 2.
            
        Returns:
            numpy.ndarray: Frame with the ball control statistics overlay.
        """
        # Create semi-transparent background rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        # Calculate ball possession statistics up to current frame
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
        # Count frames where each team had the ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        # Calculate possession percentages
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:  # Avoid division by zero
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1 = team_2 = 0.0

        # Display possession statistics
        cv2.putText(frame, f"Team 1 possession : {team_1*100:.2f}%", (1400, 900), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 possession : {team_2*100:.2f}%", (1400, 950), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Annotate a sequence of video frames with tracking visualization.
        
        Draws markers for players, referees, the ball, and displays ball possession statistics.
        
        Args:
            video_frames (list): List of video frames to annotate.
            tracks (dict): Tracking data from get_object_tracks().
            team_ball_control (numpy.ndarray): Array indicating which team has the ball in each frame.
            
        Returns:
            list: Annotated video frames.
        """
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            # Create a copy to avoid modifying original frames
            frame = frame.copy()

            # Get tracking data for current frame
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            ref_dict = tracks["referees"][frame_num]

            # Draw players with team colors and IDs
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # Default to red if team color not set
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # Mark players who have the ball with a triangle
                if player.get('has_ball', False):
                   frame = self.draw_triangle(frame, player["bbox"], (255, 0, 0)) 

            # Draw referees (yellow ellipses without IDs)
            for _, referee in ref_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
                
            # Draw ball (green triangle)
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw team ball possession statistics
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames