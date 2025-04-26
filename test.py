from train import get_video_paths, create_tf_dataset
import tensorflow as tf
from SFTNet import SFTNet

test_dir = ["pro_data0/test", "pro_data1/test"]
test_paths, test_labels = get_video_paths(test_dir, isTest = True)
test_dataset = create_tf_dataset(test_paths, test_labels, isTrain=False, num_frames=20)

strategy = tf.distribute.MirroredStrategy()
load_model_path = "model/SFTNet.weights.h5"
with strategy.scope():
    model = SFTNet()
    model.load_weights(load_model_path)
    loss, accuracy, auc = model.evaluate(test_dataset)
    print(f"Loss: {loss}")
    print(f"Acc: {accuracy}")
    print(f"AUC: {auc}")
