from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, 
    Input, LSTM, TimeDistributed, Bidirectional, Concatenate, Lambda
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

class SFTNet:
    def __init__(self, 
                 num_frames=20, 
                 num_epochs=30, 
                 base_model=EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
                 dct_base_model=MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))):
        """
        Initializes the SFTNet model.
        
        Parameters:
            base_model: A tf.keras Model, used as the base for RGB frame features.
                        In this example, it should be EfficientNetV2B0.
            num_frames: Number of frames per video.
        """
        self.num_frames = num_frames
        self.num_epochs = num_epochs
        self.base_model = base_model
        self.dct_base_model = dct_base_model

        self.base_model.trainable = True
        self.dct_base_model.trainable = True  # Set trainability as needed
        
        self.model = self._build_model()
    
    @staticmethod
    def _compute_2d_dct_frame(frame):
        """
        Compute a 2D DCT for a single frame.
        The frame shape is (H, W, C). For each channel, we apply the DCT over the width (last axis)
        and then over the height (by transposing so the height becomes the last axis).
        """
        frame = tf.cast(frame, tf.float32)
        channels = tf.unstack(frame, axis=-1)
        transformed_channels = []
        for ch in channels:
            dct_width = tf.signal.dct(ch, type=2, norm='ortho')
            transposed = tf.transpose(dct_width)
            dct_height = tf.signal.dct(transposed, type=2, norm='ortho')
            dct_ch = tf.transpose(dct_height)
            transformed_channels.append(dct_ch)
        dct_frame = tf.stack(transformed_channels, axis=-1)
        return dct_frame

    @classmethod
    def _dct_2d(cls, videos):
        """
        Processes a batch of videos where each video has shape (NUM_FRAME, H, W, C).
        Applies the 2D DCT frame-by-frame.
        """
        def process_video(video):
            # video: shape (NUM_FRAME, H, W, C)
            return tf.map_fn(lambda frame: cls._compute_2d_dct_frame(frame),
                            video, fn_output_signature=tf.float32)
        return tf.map_fn(process_video, videos, fn_output_signature=tf.float32)
    
    def _build_model(self):
        # Input for a batch of videos: (NUM_FRAME, 224, 224, 3)
        video_input = Input(shape=(self.num_frames, 224, 224, 3), name="video_input")
        
        branch_outputs = []

        # Branch 1: Extract features from RGB frames using the base EfficientNetV2B0 model.
        feature_layer_names = [
            "block2b_expand_activation",   # Change this to an appropriate layer name if needed.
            "top_activation"               # Likewise, verify the layer exists in the EfficientNetV2B0 model.
        ]
        
        for i, layer_name in enumerate(feature_layer_names):
            # Create a branch extractor model from the base model
            branch_model = Model(
                inputs=self.base_model.input,
                outputs=self.base_model.get_layer(layer_name).output,
                name=f"branch_extractor_{layer_name}"
            )

            # Process each frame using the branch extractor in a time-distributed manner
            td_features = TimeDistributed(branch_model, name=f"td_extractor_{i}")(video_input)

            # Pool spatial dimensions for each frame
            pooled = TimeDistributed(GlobalAveragePooling2D(), name=f"td_gap_branch_{i}")(td_features)
            
            # Process the temporal sequence for this branch with a Bidirectional LSTM
            lstm_out = Bidirectional(LSTM(64, return_sequences=True), name=f"bidir_lstm_branch_{i}")(pooled)
            branch_outputs.append(lstm_out)
        
        # Branch 2: Process DCT-transformed frames.
        dct_transformed = Lambda(self._dct_2d, name="video_dct_transform")(video_input)        
        dct_td_features = TimeDistributed(self.dct_base_model, name="td_mobilenet_dct")(dct_transformed)
        dct_pooled = TimeDistributed(GlobalAveragePooling2D(), name="td_gap_dct")(dct_td_features)
        dct_lstm = Bidirectional(LSTM(64, return_sequences=True), name="bidir_lstm_dct")(dct_pooled)
        branch_outputs.append(dct_lstm)
        
        # Concatenate the outputs from both branches
        stacked_tokens = Concatenate()(branch_outputs)
        
        # Fuse the temporal information from all branches
        x = Bidirectional(LSTM(128, return_sequences=False), name="all_lstm")(stacked_tokens)
        
        # Apply an MLP for classification
        x = Dense(256,
                  activation='relu',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  name="mlp_dense")(x)
        x = BatchNormalization(name="mlp_bn")(x)
        x = Dropout(0.5, name="mlp_dropout")(x)
        predictions = Dense(1, activation='sigmoid', name="predictions")(x)
        
        model = Model(inputs=video_input, outputs=predictions, name="video_classification_model")
        
        # Compile the model
        optimizer = Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        
        return model
    
    def summary(self):
        """Prints the model summary."""
        self.model.summary()

    def save_weights(self, save_path='model/SFTNet.weights.h5'):
        """Saves model weights."""
        self.model.save_weights(save_path)

    def load_weights(self, load_path='model/SFTNet.weights.h5'):
        """Loads model weights."""
        self.model.load_weights(load_path)
    
    def train(self, train_dataset, val_dataset, save_path='model/SFTNet.weights.h5'):
        """Trains the model."""
        model_checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_accuracy', mode='max', save_weights_only=True)
        history = self.model.fit(train_dataset, epochs=self.num_epochs, validation_data=val_dataset, callbacks=[model_checkpoint])
        return history
    
    def evaluate(self, test_dataset):
        """Evaluates the model."""
        return self.model.evaluate(test_dataset)
    
    