from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import sys
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) ## import the YOLO model int self.model
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
            ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
            print(ball_positions)
            df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

            #print(df_ball_positions)

            # Interpolate missing values
            df_ball_positions = df_ball_positions.interpolate()
            df_ball_positions = df_ball_positions.bfill()

            ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

            # print(ball_positions)
            return ball_positions



    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)   ## Do the predictions on video
            detections += detections_batch

        return detections


    def get_object_track(self, frames, read_from_stub=False, stub_path=None):


        ## this with see that the operations is already done or not if it is done it load the it otherwise it will run the code from starting
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
         
        tracks={                                            ## for dictionary formate
            "players":[],                                   ## {0:{"bbox":{0,0,0,0}, 1:bbox{0,0,0,0}, ........... 21:{0,0,0,0}}}        frame 0   and same for multiple frames
            "referees":[],                              
            "ball":[]

        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names                                 # {0:person, 1:goalkeeper, ...........}
            cls_names_inv = {v:k for k,v in cls_names.items()}          # {person:0, goalkeeper:1, ...........}
            print(cls_names)

            ## convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            ## convert goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            ## Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            

            ## Cnverting the supervision to dictionary
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][track_id] = {"bbox":bbox}

        # save the file
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        ## Drawing ellipse arround player and referee
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,  # Image to draw on
            (x_center, y2),  # Center of the ellipse
            (int(width), int(0.35 * width)),  # Axes lengths
            0.0,  # Angle of rotation
            -45,  # Starting angle
            235,  # Ending angle
            color,  # Color of the ellipse
            2,  # Thickness of the ellipse border
            cv2.LINE_4  # Line type
        )


        # Draw rectange that contain the track_id of the player
        rectangle_width=40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect=(y2-rectangle_height//2) + 15
        y2_rect=(y2+rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y2_rect-3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )


        return frame

    # Draw the triangle on the ball
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)    ## connect the lines
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)           ## Draw the boarder

        return frame


    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Safely get player, ball, and referee dictionaries
            player_dict = tracks.get("players", [{}])[frame_num]
            ball_dict = tracks.get("ball", [{}])[frame_num]
            referee_dict = tracks.get("referees", [{}])[frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                bbox = player.get("bbox", [])
                frame = self.draw_ellipse(frame, bbox, color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))
               
            # Draw Referees
            for _, referee in referee_dict.items():
                bbox = referee.get("bbox", [])
                frame = self.draw_ellipse(frame, bbox, (0, 255, 255))
                
            
            # Draw the ball
            for track_id, ball in ball_dict.items():
                bbox = ball.get("bbox", [])
                frame = self.draw_triangle(frame, bbox, (0, 255, 0))
                


            output_video_frames.append(frame)

        return output_video_frames

         