from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read video
    vid_frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(vid_frames,read_from_stub=True,stub_path='stubs/track_stubs.pkl')

    #Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    #Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colors(vid_frames[1],tracks["players"][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(vid_frames[frame_num],track['bbox'],player_id)

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # Assign Ball acquisition
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks["ball"][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
    

    #Draw Output
    # Draw Object Tracks
    output_video_frames = tracker.draw_annotations(vid_frames,tracks)

    # Save video
    save_video(output_video_frames,'output_videos/output_video.avi')

if __name__ == '__main__':
    main()