import os
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips


def merge_vid(folder_path, output_filename):
    # check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Get all the files path
    file_list = os.listdir(folder_path)

    # Filtering : only the .mp4 files are kept
    video_files = [f for f in file_list if f.endswith('.MP4')]

    # Sort the files by alphabelitical order
    video_files.sort()

    # double check if the folder is not empty
    if len(video_files) == 0:
        print("The folder does not contain any videos")
        return

    # determination of the video properties
    video_properties = cv2.VideoCapture(folder_path + '\\' + video_files[0])
    width = int(video_properties.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_properties.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_properties.get(cv2.CAP_PROP_FPS)
    video_properties.release()

    video_path_complete = [VideoFileClip(folder_path + '\\' + video_files[k]) for k in range(len(video_files))]

    # Merge the videos
    m_v = concatenate_videoclips(video_path_complete)
    # Resize the video
    m_v_resized = m_v.resize((640,640))
    # Save merged video with adjusted resolution
    m_v_resized.write_videofile(folder_path + '\\' + output_filename, codec="libx264")
    print(f"The videos have been successfully recomposed")

if __name__ == "__main__":

    merge_vid(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\ExampleofFlight\2019-03-25_5_LukeFuckingAmazingLoadsofAshFalling\GoPro Front", 'full_flight_640.mp4') #it is only working threre, strange
