import os
import cv2
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

save_directory = "/tmp"


def setResultsDir(c):
    global save_directory
    save_directory = c


def save_values_to_json(values_dict, json_file_path):
    # Create the full file path within the save directory
    file_path = os.path.join(save_directory, json_file_path)

    # Save the values to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(values_dict, json_file, indent=2)

def plot_accuracy_and_loss_all_history(histories):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Predefined color list
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, history in enumerate(histories):
        # Get the training accuracy and validation accuracy from the history object
        training_accuracy = history.history['accuracy']
        validation_accuracy = history.history['val_accuracy']

        # Get the training loss and validation loss from the history object
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']
        epochs = range(1, len(training_accuracy) + 1)

        # Get the color for this run from the predefined list
        color = color_list[i % len(color_list)]

        # Plot the training accuracy on the first subplot
        ax1.plot(epochs, training_accuracy,color, label=f'Training acc. - RunTime {i+1}')
        ax1.plot(epochs, validation_accuracy,color+'--', label=f'Validation acc. - RunTime {i+1}')

        # Plot the training loss on the second subplot
        ax2.plot(epochs, training_loss, label=f'Training loss - RunTime {i+1}')
        ax2.plot(epochs, validation_loss, label=f'Validation loss - RunTime {i+1}')

    ax1.set_title('Accuracy: Training vs Validation', fontsize=20)
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.legend(fontsize=12)

    ax2.set_title('Loss: Training vs Validation', fontsize=20)
    ax2.set_xlabel('Epochs', fontsize=16)
    ax2.set_ylabel('Loss', fontsize=16)
    ax2.legend(fontsize=12)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    # Save the plot as an EPS file
    fig.savefig(os.path.join(save_directory, 'accuracy_and_loss_plot.eps'), format='eps')

    # Save the values as a JSON file
    values_dict = {'accuracy_and_loss': {}}
    for i, history in enumerate(histories):
        values_dict['accuracy_and_loss'][f'RunTime_{i+1}'] = {
            'training_accuracy': history['training_accuracy'],
            'validation_accuracy': history['validation_accuracy'],
            'training_loss': history['training_loss'],
            'validation_loss': history['validation_loss']
        }
    save_values_to_json(values_dict, 'accuracy_and_loss_values.json')


def plot_accuracy_and_loss(training_accuracy, validation_accuracy, training_loss, validation_loss):
    epochs = range(1, len(training_accuracy) + 1)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the training accuracy on the first subplot
    ax1.plot(epochs, training_accuracy, 'b', label='Training acc.')
    ax1.plot(epochs, validation_accuracy, 'r', label='Validation acc.')
    ax1.set_title('Accuracy: Training vs Validation', fontsize=20)
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.legend(fontsize=16)

    # Plot the training loss on the second subplot
    ax2.plot(epochs, training_loss, 'b', label='Training loss')
    ax2.plot(epochs, validation_loss, 'r', label='Validation loss')
    ax2.set_title('Loss:  Training vs Validation', fontsize=20)
    ax2.set_xlabel('Epochs', fontsize=16)
    ax2.set_ylabel('Loss', fontsize=16)
    ax2.legend(fontsize=16)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    # Save the plot as an EPS file
    fig.savefig(os.path.join(save_directory, 'accuracy_and_loss_plot.eps'), format='eps')

    # Save the values as a JSON file
    values_dict = {
        'accuracy_and_loss': {
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy,
            'training_loss': training_loss,
            'validation_loss': validation_loss
        }
    }
    save_values_to_json(values_dict, 'accuracy_and_loss_values.json')


def overlay(video, save_path):
    # Preprocess the video frames
    stacked_frames = tf.stack(list(video), axis=-1)
    average_intensity = tf.reduce_mean(stacked_frames, axis=-1)
    average_intensity_np = average_intensity.numpy()
    output_image = np.uint8(average_intensity_np * 255)
    output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)
    enhanced_image_gray = cv2.equalizeHist(output_image_gray)
    enhanced_image = cv2.cvtColor(enhanced_image_gray, cv2.COLOR_GRAY2RGB)

    # Save the preprocessed image
    cv2.imwrite(save_path, enhanced_image)

    # Return the enhanced image
    return enhanced_image


def plot_bar_chart(probabilities, save_path, video_path):
    # Create a figure with the specified size
    fig, ax = plt.subplots(figsize=(2, 6))

    # Create a bar plot using the exact format of the plot_value_array function
    ax.grid(False)
    ax.set_xticks(range(2))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    thisplot = ax.bar(range(2), [probabilities, 1 - probabilities], color=["green", "red"])
    ax.set_ylim([0, 1])

    # Update the values of the bars
    thisplot[0].set_height(probabilities)
    thisplot[1].set_height(1 - probabilities)

    # Save the plot to the specified path
    fig.savefig(save_path)

    # Close the plot
    plt.close(fig)

    # Save the values as a JSON file
    values_dict = {
        'bar_chart_values': {
            'probabilities': probabilities.tolist(),
            'save_path': save_path,
            'video_path': video_path
        }
    }
    save_values_to_json(values_dict, save_path[:-4] + '_values.json')


def plot_predictions(predictions, test_videos_tensor, test_vid_paths):
    # Convert the test_videos_tensor to a NumPy array
    test_videos_array = np.array(list(test_videos_tensor.as_numpy_iterator()))

    # Convert the test_videos_array to a list
    test_videos_list = test_videos_array.tolist()

    # Convert the predictions list to a TensorFlow tensor
    predictions_tensor = tf.convert_to_tensor(predictions)

    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(predictions_tensor)

    # Convert probabilities to a NumPy array
    probabilities_array = probabilities.numpy()

    # Normalize probabilities to ensure they total to 100%
    normalized_probabilities = probabilities_array / np.sum(probabilities_array, axis=1, keepdims=True)

    # Convert the probabilities to float32
    normalized_probabilities = normalized_probabilities.astype(np.float32)

    # Convert the probabilities to a list
    probabilities_list = normalized_probabilities.tolist()

    # Find the indices of the top ten healthy and ill probabilities
    top_healthy_indices = np.argsort(probabilities_list, axis=0)[-10:, 1]
    top_ill_indices = np.argsort(probabilities_list, axis=0)[-10:, 0]

    # Extract the videos with the top ten healthy probabilities
    top_healthy_videos = []
    top_healthy_paths = []

    for i in top_healthy_indices:
        top_healthy_videos.append(test_videos_list[i])
        top_healthy_paths.append(test_vid_paths[i])

    # Extract the videos with the top ten ill probabilities
    top_ill_videos = []
    top_ill_paths = []

    for i in top_ill_indices:
        top_ill_videos.append(test_videos_list[i])
        top_ill_paths.append(test_vid_paths[i])

    # Preprocess the videos and save the images
    for i, video in enumerate(top_healthy_videos):
        save_path = os.path.join(save_directory, f'{i + 1}_healthy_overlay.png')
        overlay(video, save_path)

    for i, video in enumerate(top_ill_videos):
        save_path = os.path.join(save_directory, f'{i + 1}_ill_overlay.png')
        overlay(video, save_path)

    # Get the probabilities of being healthy and ill for the selected videos
    top_healthy_probabilities = normalized_probabilities[top_healthy_indices][:, 1]
    top_ill_probabilities = normalized_probabilities[top_ill_indices][:, 0]

    # Create the bar charts for the top ten healthy videos
    for i in range(len(top_healthy_videos)):
        plot_bar_chart(top_healthy_probabilities[i], os.path.join(save_directory, f'{i + 1}_healthy_bar_chart.eps'),
                       top_healthy_paths[i])

    # Create the bar charts for the top ten ill videos
    for i in range(len(top_ill_videos)):
        reversed_probability = 1 - top_ill_probabilities[i]
        plot_bar_chart(reversed_probability, os.path.join(save_directory, f'{i + 1}_ill_bar_chart.eps'),
                       top_ill_paths[i])
