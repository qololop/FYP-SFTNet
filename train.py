from SFTNet import SFTNet
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import re
import random
import os

def get_video_paths(base_dirs, isTest = False):
    """
    Get video paths and labels from the specified directories.
    """

    video_paths = []
    labels = []

    num_fake = 0
    num_real = 0

    for base_dir in base_dirs:
        class_paths = [os.path.join(base_dir, "fake"), os.path.join(base_dir, "real")]
        num_fake += len(os.listdir(class_paths[0]))
        num_real += len(os.listdir(class_paths[1]))
    times = num_fake // num_real

    for base_dir in base_dirs:
        for label, folder in enumerate(["fake", "real"]):
            class_path = os.path.join(base_dir, folder)
    
            if folder == "real" or isTest:
                video_names = sorted(os.listdir(class_path))
            else:
                video_names = sorted(os.listdir(class_path))[::times]
            
            for video_name in video_names:
                video_path = os.path.join(class_path, video_name)
                
                if os.path.isdir(video_path):  
                    video_paths.append(video_path)
                    labels.append(label)  # 1 for real, 0 for fake
                else:
                    print(video_path)
        
    video_paths = np.array(video_paths)
    labels = np.array(labels)

    temp = list(zip(video_paths, labels))
    random.shuffle(temp)
    res1, res2 = zip(*temp)

    video_paths, labels = list(res1), list(res2)

    return video_paths, labels

def load_video_frames(video_path, num_frames=10, img_size=(224,224)):
    """
    Load video frames from a directory.
    """

    video_path = str(video_path)
    
    def extract_frame_number(filename):
        match = re.findall(r'[0-9]+', filename)  # Extract all numbers
        return int(match[0]) if match else float('inf')
        
    frame_files = sorted(
        [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))],
        key=extract_frame_number
    )[:num_frames]

    frame_paths = [os.path.join(video_path, f) for f in frame_files if f.endswith(('.jpg', '.png'))]
    
    # Select first 'num_frames' frames
    frames = []
    for i in range(num_frames):
        if i < len(frame_paths):
            img = image.load_img(frame_paths[i], target_size=img_size)
            img_arr = image.img_to_array(img)
            img_arr = img_arr / 255.0
            frames.append(img_arr)
        else:
            frames.append(np.zeros((*img_size, 3)))  # Pad with black frames if not enough
    frames = np.array(frames)
    return frames

def video_generator(video_paths, video_labels, num_frames=10):
    """
    Generator function to load videos dynamically from disk.
    """
    for video_path, label in zip(video_paths, video_labels):
        video_path = str(video_path)
        frames = load_video_frames(video_path, num_frames)
        yield frames.astype(np.float32), np.array(label, dtype=np.int32)

def create_tf_dataset(video_paths, video_labels, num_frames=10, batch_size=8, isTrain=True):
    """
    Creates a `tf.data.Dataset` for efficient video loading.
    """
    output_signature = (
        tf.TensorSpec(shape=(num_frames, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: video_generator(video_paths, video_labels, num_frames=num_frames),
        output_signature=output_signature
    )

    # Shuffle, batch, and prefetch
    if isTrain:
        dataset = dataset.shuffle(50)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":        
    train_dir = ["pro_data0/train", "pro_data1/train"]
    val_dir = ["pro_data0/val", "pro_data1/val"]

    train_paths, train_labels = get_video_paths(train_dir, isTest = False)
    val_paths, val_labels = get_video_paths(val_dir, isTest = True)

    train_dataset = create_tf_dataset(train_paths, train_labels, isTrain=True, num_frames=20)
    val_dataset = create_tf_dataset(val_paths, val_labels, isTrain=False, num_frames=20)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = SFTNet()
        model.summary()
        model.train(train_dataset, val_dataset)