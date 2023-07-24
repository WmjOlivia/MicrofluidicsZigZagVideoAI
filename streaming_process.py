import os
import sys
import logging

import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.applications import EfficientNetB0
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Rescaling, TimeDistributed, Dense, GlobalAveragePooling3D, Dropout

import cv2
import numpy as np

import plots
from video_processor import get_videos, save_video_labels_to_file, process_dataset

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
        # 跳过无法读取的帧
        if not ret or frame.size == 0:
            break
        # 应用背景减除到帧上
        fgMask = bgSub.apply(frame)
        # 将前景掩模转换为RGB
        mask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
        frame = frame / 255.0
        # 将掩模应用到帧上
        preprocessed_frame = cv2.bitwise_and(frame, mask)
        if ret:
            # preprocessed_frame = preprocess_frame(frame)
            frames.append(preprocessed_frame)
    cap.release()
     # 如果视频中没有帧，则跳过该视频
    if len(frames) == 0:
        return
    frames = np.array(frames)[..., [2, 1, 0]]
    frames = np.maximum(frames, 0)
    frames = tf.convert_to_tensor(frames)
    predictions = base_model.predict(frames)
    del(frames)
    return predictions

# 生成器函数
def data_generator(video_paths, labels, batch_size, num_frames, base_model):
    num_samples = len(video_paths)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    start_index = 0
    while True:
        batch_indices = indices[start_index:start_index+batch_size]
        batch_video_paths = [video_paths[i] for i in batch_indices]
        batch_labels = labels[batch_indices]
        batch_data = []
        for video_path in batch_video_paths:
            predictions = predict_video_frames(video_path, num_frames, base_model)
            batch_data.append(predictions)
        batch_data = np.array(batch_data)
        batch_data = np.reshape(batch_data, (len(batch_data), -1))
        yield batch_data, batch_labels
        start_index += batch_size
        if start_index >= num_samples:
            start_index = 0
