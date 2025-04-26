import os
import shutil
import av
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def extract_and_save_frames(video_path, output_size=(224, 224), step=10, output_dir="pro_data/train/real", conf_threshold=0.5):
    """
    Extract 1 frame every `step` frames from `video_path` and save as JPEGs in a 
    subdirectory (named after the video) under `output_dir`.
    
    Args:
        video_path (str): Path to the input video file.
        output_size (tuple): The desired (width, height) for the frame images.
        step (int): Extract one frame every this many frames.
        output_dir (str): The parent directory where the subdirectory for this video will be created.
    """
    # Prepare output subdirectory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)

    if os.path.exists(video_output_dir):
        return
    
    os.makedirs(video_output_dir, exist_ok=True)

    try:
        # Load the YOLO face detection model
        model = YOLO(YOLO_path)
        
        # Open video file with PyAV
        container = av.open(video_path)
        
        frame_index = 0  # count of processed frames in video
        saved_frame_count = 0  # count of saved frames
    
        # Decode video frames from the container
        for frame in container.decode(video=0):
            if frame_index % step == 0:
                # Convert to a numpy array (BGR format)
                img = frame.to_ndarray(format="bgr24")
    
                # Run the YOLO face detection on the image
                result = model(img, verbose=False)
    
                # Only collect the first detection face
                face_crop = None
                if result and len(result) > 0 and result[0].boxes is not None:
                    for box in result[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf.cpu().numpy()
    
                        x1 -= 70
                        x2 += 70
                        y1 -= 55
                        y2 += 20
    
                        if conf >= conf_threshold:
                            h, w = img.shape[:2]
                            x1 = max(0, int(x1))
                            y1 = max(0, int(y1))
                            x2 = min(w, int(x2))
                            y2 = min(h, int(y2))
    
                            face_crop = img[y1:y2, x1:x2]
                            break
    
                if face_crop is not None and face_crop.size != 0:
                    # Optionally resize the frame
                    resized = cv2.resize(face_crop, output_size)
                    # Save image as jpg
                    output_filename = os.path.join(video_output_dir, f"frame_{saved_frame_count:04d}.jpg")
                    cv2.imwrite(output_filename, resized)
                    #print(f"Saved {output_filename}")
    
                    saved_frame_count += 1
            
            frame_index += 1
            if saved_frame_count == 20:
                break
    
        container.close()
        #print(f"Extraction complete. Saved {saved_frame_count} frames in {video_output_dir}")
    
    except Exception as e:
        if os.path.exists(video_output_dir):
            shutil.rmtree(video_output_dir)
        print(e)
        return

def process_videos(folder_path, output_path):
    for video_file in tqdm(os.listdir(folder_path)):
        video_path = os.path.join(folder_path, video_file)
        extract_and_save_frames(video_path, output_dir=output_path)

if __name__ == "__main__":
    YOLO_path = "yolov11s-face.pt"
    for i in ["train", "val", "test"]:
        for j in ["real", "fake"]:
            process_videos(f"data/{i}/{j}", f"pro_data0/{i}/{j}")
    for i in ["train", "val", "test"]:
        for j in ["real", "fake"]:
            process_videos(f"Celeb/{i}/{j}", f"pro_data1/{i}/{j}")
    
    for i in ["pro_data0", "pro_data1"]:
        for j in ["train", "val", "test"]:
            for k in ["fake", "real"]:
                allfile = f"{i}/{j}/{k}"
                for path in os.listdir(allfile):
                    full_path = os.path.join(allfile, path)
                    if len(os.listdir(full_path)) < 20:
                        shutil.rmtree(full_path)

