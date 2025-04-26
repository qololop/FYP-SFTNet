import json
import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = "FF++"

paths = [os.path.join(base_dir, "train.json"), os.path.join(base_dir, "val.json"), os.path.join(base_dir, "test.json")]
with open(paths[0], "r") as f:
    train = json.load(f)
    train = [x for l in train for x in l]
with open(paths[1], "r") as f:
    val = json.load(f)
    val = [x for l in val for x in l]
with open(paths[2], "r") as f:
    test = json.load(f)
    test = [x for l in test for x in l]

original_dir = os.path.join(base_dir, "original")
NT_dir = os.path.join(base_dir, "NT")
F2F_dir = os.path.join(base_dir, "F2F")  
FS_dir = os.path.join(base_dir, "FS")
FSw_dir = os.path.join(base_dir, "FSw")
DF_dir = os.path.join(base_dir, "DF")
man_dirs = [NT_dir, F2F_dir, FS_dir, FSw_dir, DF_dir]

YOLO_path = "yolov11s-face.pt"

for x in ["train", "val", "test"]:
    for y in ["real", "fake"]:
        os.makedirs(f"data/{x}/{y}")

for x in ["train", "val", "test"]:
    for y in ["real", "fake"]:
        os.makedirs(f"pro_data/{x}/{y}")

allfiles = os.listdir(original_dir)
for f in allfiles:
    src = os.path.join(original_dir, f)
    name = f.split(".")[0]
    if name in train:
        dest = "data/train/real"
    elif name in val:
        dest = "data/val/real"
    elif name in test:
        dest = "data/test/real"
    shutil.move(src, dest)  

n = 0
for df_dir in man_dirs:
    allfiles = os.listdir(df_dir)
    for f in allfiles:
        src = os.path.join(df_dir, f)
        name = f.split(".")[0].split("_")[0]
        fn = name + f"_{n}.mp4"
        if name in train:
            dest = os.path.join("data/train/fake", fn)
        elif name in val:
            dest = os.path.join("data/val/fake", fn)
        elif name in test:
            dest = os.path.join("data/test/fake", fn)
        os.rename(src, dest)
    n += 1      

# CelebDF-v2
celeb_base = "celeb-df-v2"
celeb_test_list = "celeb-df-v2/List_of_testing_videos.txt"
with open(celeb_test_list) as f:
    see = f.readlines()

for x in ["train", "val", "test"]:
    for y in ["real", "fake"]:
        os.makedirs(f"Celeb/{x}/{y}")

for y in ["real", "fake"]:
    os.makedirs(f"pre_data/{y}")

celeb_real_path = os.path.join(celeb_base, "Celeb-real")
youtube_real_path = os.path.join(celeb_base, "YouTube-real")
celeb_syn_path = os.path.join(celeb_base, "Celeb-synthesis")

for vid in os.listdir(celeb_real_path):
    src = os.path.join(celeb_real_path, vid)
    dest = os.path.join("pre_data/real", vid)
    shutil.copyfile(src, dest)
for vid in os.listdir(youtube_real_path):
    src = os.path.join(youtube_real_path, vid)
    dest = os.path.join("pre_data/real", vid)
    shutil.copyfile(src, dest)
for vid in os.listdir(celeb_syn_path):
    src = os.path.join(celeb_syn_path, vid)
    dest = os.path.join("pre_data/fake", vid)
    shutil.copyfile(src, dest)

for line in see:
    label, path = line.split(" ")
    path = path.strip().split("/")[1]
    if label == "0":
        src = os.path.join("pre_data/fake", path)
        dest = os.path.join("Celeb/test/fake", path)
    elif label == "1":
        src = os.path.join("pre_data/real", path)
        dest = os.path.join("Celeb/test/real", path)
    shutil.move(src, dest)
 
video_paths = os.listdir("pre_data/real")
tmp = [1 for _ in range(len(video_paths))]
video_paths.extend(os.listdir("pre_data/fake"))
tmp.extend([0 for _ in range(len(video_paths)-len(tmp))])


train_paths, val_paths, train_labels, val_labels = train_test_split(
    video_paths, tmp, test_size=0.2, random_state=42, stratify=tmp, shuffle=True
)

for path, label in zip(train_paths, train_labels):
    if label == 0:
        src = f"pre_data/fake/{path}"
        dest = f"Celeb/train/fake/{path}"
    elif label == 1:
        src = f"pre_data/real/{path}"
        dest = f"Celeb/train/real/{path}"
    shutil.move(src,dest)

for path, label in zip(val_paths, val_labels):
    if label == 0:
        src = f"pre_data/fake/{path}"
        dest = f"Celeb/val/fake/{path}"
    elif label == 1:
        src = f"pre_data/real/{path}"
        dest = f"Celeb/val/real/{path}"
    shutil.move(src,dest)

shutil.rmtree("pre_data")

for x in ["train", "val", "test"]:
    for y in ["real", "fake"]:
        os.makedirs(f"pro_data1/{x}/{y}")