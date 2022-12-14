#!/usr/bin/python3

import tqdm
import random
import pathlib
import itertools
import collections
import copy

import os
import cv2
import numpy as np
import tensorflow as tf

class AVIfile:

  def __init__(self, video_path, label_name, clip_length, crop_rect = False, frame_step = 1, frames2ret = False, subtract_background = False):
    """
    Wrapper for the raw AVI file.
    video_path: path to the AVI video clip
    label_name: label of the dataset, for example healthy or ill
    clip_length: the number of frames per clip
    crop_rect: rectangle to crop the frames
    frame_step: defines how many frames to skip to reduce the amount of data
    frames2ret: frames per clip to return if not the whole clip_length is taken
    subtract_background: if background subtraction should be performed or not
    """
    self.clip_length = clip_length
    self.label_name = label_name
    self.crop_rect = crop_rect
    self.frame_step = frame_step
    self.subtract_background = subtract_background
    if not frames2ret:
      self.frames2ret = clip_length
    else:
      self.frames2ret = frames2ret
    self.src = cv2.VideoCapture(str(video_path))
    if not self.src:
      print("Could not open:",video_path)
    self.video_length = int(self.src.get(cv2.CAP_PROP_FRAME_COUNT))
    self.width  = int(self.src.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    self.height = int(self.src.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    print("{} opened: {} frames, {} clips at {}x{}, bgsub={}".format(video_path,self.video_length,self.get_number_of_clips(),
    self.width,self.height,self.subtract_background))

  def calcBackground(self,allframes):
    # at the momemnt just naive average
    avg = np.zeros_like(allframes[0])
    for f in allframes:
      avg += f
    avg = avg / len(allframes)
    return avg

  def subtractBackground(self, allframes):
    background = self.calcBackground(allframes)
    frames_without_bg = []
    for f in allframes:
      if self.subtract_background in "abs":
        f2 = np.abs(f - background)
      elif self.subtract_background in "rect":
        f2 = np.maximum(f - background, 0)
      else:
        print("Undefined backgroud subtraction method")
        quit()
      frames_without_bg.append(f2)
    return frames_without_bg

  def __del__(self):
    self.src.release()

  def format_frames(self,frame):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    return frame

  def get_number_of_clips(self):
    return self.video_length // self.clip_length

  def get_frames_of_clip(self, clip_index):
    output_size = (self.width, self.height)

    framepos =  clip_index * self.clip_length
    print("---> {} going to frame pos {}, clip index = {}.".
      format(self.label_name,framepos,clip_index))
    self.src.set(cv2.CAP_PROP_POS_FRAMES,framepos)

    result = []

    ret, frame = self.src.read()
    if not ret:
      print("Frame read error")
      quit()
    frame = self.format_frames(frame)

    result.append(frame)

    for i in range(1,self.frames2ret):
      ret, frame = self.src.read()
      if (i % self.frame_step) == 0:
        if ret:
          frame = self.format_frames(frame)
          result.append(frame)
        else:
          result.append(np.zeros_like(result[0]))
    ## returning the array but fixing openCV's odd BGR format to RGB
    result = np.array(result)[...,[2,1,0]]

    if self.crop_rect:
      result = tf.image.crop_to_bounding_box(result, 
      self.crop_rect[0][1], self.crop_rect[0][0], 
      self.crop_rect[1][1]-self.crop_rect[0][1], self.crop_rect[1][0]-self.crop_rect[0][0])

    if self.subtract_background:
      result = self.subtractBackground(result)

    return result


class FrameGenerator:
  """
  Class which feeds a list of label/video into the TF film classifier. Done as a stream.
  """
  def __init__(self, avifiles, clips_list, training = False):
    self.avifiles = avifiles
    self.training = training
    self.clips_list = clips_list

  def __call__(self):
    pairs = []
    for clip_index in self.clips_list:
      for label_index in range(len(self.avifiles)):
        pair = (label_index,clip_index)
        pairs.append(pair)

    if self.training:
      random.shuffle(pairs)

    for label_index,clip_index in pairs:
      video_frames = self.avifiles[label_index].get_frames_of_clip(clip_index)
      label = label_index
      yield video_frames, label
