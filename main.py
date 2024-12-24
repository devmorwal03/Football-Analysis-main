from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    video_frame = read_video('08fd33_4.mp4')

    # initialize the tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_track(video_frame, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Interpolate ball positions
    # tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frame[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frame[frame_num], track["bbox"], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.Team_colors[team]

            
    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True

    
    # Draw output
    # Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frame, tracks)

    # save video
    save_video(output_video_frames, 'output_video/output_video.avi')

if __name__ == '__main__':
    main()