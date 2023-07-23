import logging
from tqdm import tqdm
import cv2
import random
import numpy as np
import os
import tensorflow as tf


def get_videos(path_list, label, num_videos):
    # Initialize variables.
    video_paths = []
    video_labels = []
    all_videos = []

    # Iterate over the paths.
    for path in path_list:
        # Get a list of video paths in the directory.
        videos = [os.path.join(path, f)
                  for f in os.listdir(path)
                  if f.endswith('.avi') and os.path.isfile(os.path.join(path, f))]

        # Add the video paths to the list of all videos.
        all_videos.extend(videos)

    # Shuffle the list of all videos.
    random.shuffle(all_videos)

    # Get the first `num_videos` videos from the list of all videos.
    selected_videos = all_videos[:num_videos]

    # Iterate over the selected videos.
    for video in selected_videos:
        # Add the video path and label to the lists of video paths and labels.
        video_paths.append(video)
        video_labels.append(label)

    # Return the lists of video paths and labels.
    return video_paths, video_labels


def save_video_labels_to_file(filename, video_paths, labels):
    with open(filename, "w") as file:
        for video_path, label in zip(video_paths, labels):
            file.write(f"{video_path},{label}\n")


def process_dataset(native_videos, modified_videos, native_labels, modified_labels):
    # Process the native videos.
    processed_native_videos, native_videos_paths = process_videos(native_videos)
    processed_modified_videos, modified_videos_paths = process_videos(modified_videos)

    # Concatenate the native and modified data.
    processed_videos = np.concatenate([processed_native_videos, processed_modified_videos], axis=0)
    processed_videos = processed_videos.astype(np.float32) / 255.0
    processed_videos = tf.data.Dataset.from_tensor_slices(processed_videos)
     # Use the provided labels.
    labels = np.concatenate([native_labels, modified_labels], axis=0)
    labels = labels.astype(np.int16)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    all_video_paths = native_videos_paths + modified_videos_paths

    # Return the processed data, labels, and video paths.
    return processed_videos, labels, all_video_paths


def process_videos(videos):
    # Initialize variables.
    processed_videos = []
    video_paths = []
    videos_id_list = []
    bgSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)

    # Iterate over the video paths.
    for video_path in tqdm(videos, desc='Processing videos', position=0, leave=True):
        # Open the video file.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Could not open video file: {video_path}")
            continue

        # Get the number of frames in the video.
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize the video frames list.
        video_frames = []

        # Iterate over the frames in the video.
        for frame_count in range(num_frames):
            ret, frame = cap.read()

            # Skip the frame if it could not be read.
            if not ret or frame.size == 0:
                break

            # Apply background subtraction to the frame.
            fgMask = bgSub.apply(frame)

            # Convert the foreground mask to RGB.
            mask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)

            # Apply the mask to the frame.
            processed_frame = cv2.bitwise_and(frame, mask)

            # Skip the first and last 10 frames.
            if 50 < frame_count <= 150:
                # Skip every other frame.
                if frame_count % 2 == 0:
                    video_frames.append(processed_frame)

        # Close the video file.
        cap.release()

        # If there are no frames in the video, skip it.
        if len(video_frames) == 0:
            continue

        # Store the video path and label
        video_paths.append(video_path)

        # Stack the frames into a NumPy array.
        video_frames = np.array(video_frames)[..., [2, 1, 0]]
        video_frames = np.maximum(video_frames, 0)
        processed_videos.append(np.stack(video_frames, axis=0))
        del video_frames
    # Return the processed videos, video paths, and their labels.
    return processed_videos, video_paths

# Processing videos and frames, obtaining datasets
def get_dataset(native_videos, modified_videos, native_labels, modified_labels, start_index, train_index, val_index, log_directory):
    # Split the videos and labels into train, validation, and test sets.
    train_native_videos, train_native_labels = native_videos[start_index:train_index], native_labels[start_index:train_index]
    train_modified_videos, train_modified_labels = modified_videos[start_index:train_index], modified_labels[start_index:train_index]

    val_native_videos, val_native_labels = native_videos[train_index:val_index], native_labels[train_index:val_index]
    val_modified_videos, val_modified_labels = modified_videos[train_index:val_index], modified_labels[train_index:val_index]

    # test_native_videos, test_native_labels = native_videos[val_index:test_index], native_labels[val_index:test_index]
    # test_modified_videos, test_modified_labels = modified_videos[val_index:test_index], modified_labels[val_index:test_index]

    # Save videos and labels
    save_video_labels_to_file(os.path.join(log_directory, "train_videos.txt"), train_native_videos + train_modified_videos,
                              train_native_labels + train_modified_labels)
    save_video_labels_to_file(os.path.join(log_directory, "val_videos.txt"), val_native_videos + val_modified_videos,
                              val_native_labels + val_modified_labels)
    # save_video_labels_to_file(os.path.join(log_directory, "test_videos.txt"), test_native_videos + test_modified_videos,
    #                           test_native_labels + test_modified_labels)

    # Split the dataset into train, validation, and test sets.
    train_videos_tensor, train_labels_tensor, train_vid_paths = process_dataset(train_native_videos,
                                                                                train_modified_videos,
                                                                                train_native_labels,
                                                                                train_modified_labels)
    val_videos_tensor, val_labels_tensor, val_vid_paths = process_dataset(val_native_videos,
                                                                          val_modified_videos,
                                                                          val_native_labels,
                                                                          val_modified_labels)
    # test_videos_tensor, test_labels_tensor, test_vid_paths = process_dataset(test_native_videos,
    #                                                                          test_modified_videos,
    #                                                                          test_native_labels,
    #                                                                          test_modified_labels)

    # Process the dataset into a form that can be used by the model
    autotune = tf.data.experimental.AUTOTUNE

    train_dataset = tf.data.Dataset.zip((train_videos_tensor, train_labels_tensor))
    train_dataset.cache().shuffle(10).prefetch(buffer_size=autotune)
    train_dataset = train_dataset.batch(1)

    val_dataset = tf.data.Dataset.zip((val_videos_tensor, val_labels_tensor))
    val_dataset.cache().prefetch(buffer_size=autotune)
    val_dataset = val_dataset.batch(1)

    # test_dataset = tf.data.Dataset.zip((test_videos_tensor, test_labels_tensor))
    # test_dataset.cache().prefetch(buffer_size=autotune)
    # test_dataset = test_dataset.batch(1)
    
    
    del(train_videos_tensor)
    del(train_labels_tensor)

    del(val_videos_tensor)
    del(val_labels_tensor)

    # del(test_labels_tensor)


    return train_dataset, val_dataset #, test_dataset, test_videos_tensor, test_vid_paths


# Processing test videos and frames, obtaining test_datasets
def get_test_dataset(native_videos, modified_videos, native_labels, modified_labels, start_index, test_index, log_directory):  
    test_native_videos, test_native_labels = native_videos[start_index:test_index], native_labels[start_index:test_index]
    test_modified_videos, test_modified_labels = modified_videos[start_index:test_index], modified_labels[start_index:test_index]

    # Save videos and labels
    save_video_labels_to_file(os.path.join(log_directory, "test_videos.txt"), test_native_videos + test_modified_videos,
                              test_native_labels + test_modified_labels)

    # Split the dataset into train, validation, and test sets.
    test_videos_tensor, test_labels_tensor, test_vid_paths = process_dataset(test_native_videos,
                                                                             test_modified_videos,
                                                                             test_native_labels,
                                                                             test_modified_labels)

    # Process the dataset into a form that can be used by the model
    autotune = tf.data.experimental.AUTOTUNE

    test_dataset = tf.data.Dataset.zip((test_videos_tensor, test_labels_tensor))
    test_dataset.cache().prefetch(buffer_size=autotune)
    test_dataset = test_dataset.batch(1)
    
    del(test_labels_tensor)


    return test_dataset, test_videos_tensor, test_vid_paths