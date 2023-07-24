import logging

import tensorflow as tf

import cv2
import numpy as np


def predict_video_frames(video_path, num_frames, base_model):
    bgSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
            logging.warning(f"Could not open video file: {video_path}")
            return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
    frames = []
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        # Skip unreadable frames
        if not ret or frame.size == 0:
            break
        # Apply background subtraction to the frame
        fgMask = bgSub.apply(frame)
        # Convert mask to RGB
        mask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
        frame = frame / 255.0
        # apply the mask to the frame
        preprocessed_frame = cv2.bitwise_and(frame, mask)
        if ret:
            # preprocessed_frame = preprocess_frame(frame)
            frames.append(preprocessed_frame)
    cap.release()
     # Skip the video if there are no frames in the video
    if len(frames) == 0:
        return
    frames = np.array(frames)[..., [2, 1, 0]]
    frames = np.maximum(frames, 0)
    frames = tf.convert_to_tensor(frames)
    predictions = base_model.predict(frames)
    # release resources
    del(frames)
    return predictions

# generator function
def data_generator(video_paths, labels, batch_size, num_frames, base_model):
    # Get the total number of samples in the dataset.
    num_samples = len(video_paths)
    
    # Create an array of indices representing the samples.
    indices = np.arange(num_samples)
    
    # Randomly shuffle the indices to introduce randomness during training.
    np.random.shuffle(indices)
    
    # Initialize the starting index to 0.
    start_index = 0
    
    # Start an infinite loop to keep generating batches indefinitely.
    while True:
        # Calculate the indices for the current batch.
        batch_indices = indices[start_index:start_index + batch_size]
        
        # Get the paths of videos corresponding to the current batch.
        batch_video_paths = [video_paths[i] for i in batch_indices]
        
        # Get the corresponding labels for the current batch.
        batch_labels = labels[batch_indices]
        
        # Initialize an empty list to store the batch data.
        batch_data = []
        
        # Loop through each video in the current batch.
        for video_path in batch_video_paths:
            # Predict frames of the video using the base_model.
            predictions = predict_video_frames(video_path, num_frames, base_model)
            
            # Append the predictions to the batch_data list.
            batch_data.append(predictions)
        
        # Convert the batch_data list to a numpy array.
        batch_data = np.array(batch_data)
        
        # Reshape the batch_data to have the shape (batch_size, num_frames * feature_dim).
        # The feature_dim represents the dimensions of the predicted frames from the base_model.
        batch_data = np.reshape(batch_data, (len(batch_data), -1))
        
        # Yield the batch data and labels to the model for training.
        yield batch_data, batch_labels
        
        # Move the start_index to the next batch.
        start_index += batch_size
        
        # If the start_index exceeds the total number of samples, start over from the beginning.
        if start_index >= num_samples:
            start_index = 0

def use_streaming(train_videos, train_labels, val_videos, val_labels, test_videos, test_labels, batch_size, num_frames)
    # train_videos = np.concatenate([train_native_videos, train_modified_videos], axis=0)
    # val_videos = np.concatenate([val_native_videos, val_modified_videos], axis=0)
    # test_videos = np.concatenate([test_native_videos, test_modified_videos], axis=0)
    # train_labels = np.concatenate([train_native_labels, train_modified_labels], axis=0)
    # val_labels = np.concatenate([val_native_labels, val_modified_labels], axis=0)
    # test_labels = np.concatenate([test_native_labels, test_modified_labels], axis=0)
    # -+=-+=-+=-+=-+=-+=-+=-+=-+=-+=-+=-+=In the main function, the above lines need to be executed first

    train_generator = data_generator(train_videos, train_labels, batch_size, num_frames)
    val_generator = data_generator(val_videos, val_labels, batch_size, num_frames)
    test_generator = data_generator(test_videos, test_labels, batch_size, num_frames)
    return train_generator, val_generator, test_generator
    # call fit:
    # model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=epochs,
    #             validation_data=val_generator, validation_steps=val_steps_per_epoch, callbacks=[csv_logger])