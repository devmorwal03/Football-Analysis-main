## This is use to read the utility and save the video

import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:                 ## ret is a flag that reads wheather the video is end or not 
            break

        frames.append(frame)

    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))  

    for frame in output_video_frames:
        out.write(frame)  

    out.release()  
