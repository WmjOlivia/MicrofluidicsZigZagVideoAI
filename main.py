#!/usr/bin/python3

import os
import sys
import logging

import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.applications import EfficientNetB0
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Rescaling, TimeDistributed, Dense, GlobalAveragePooling3D, Dropout

import plots
from video_processor import get_videos, get_test_dataset, fit_once, test_one


def logPrint(msg):
    logging.info(msg)
    print(msg)


def main():
    
    videos = 50 # video num each time
    epochs = 10 # epoch
    video_index = 1200 # FA 1450 /DA 1230  
    num_repetitions = 20
    # logPrint("repeat " + str(num_repetitions) + " times")
    if len(sys.argv) < 2:
        print("Usage: {} FA or DA or GA or MIX [-q]".format(sys.argv[0]))
        quit(0)

    option = sys.argv[1]

    # Create the results directory path
    log_directory = os.path.join(os.getcwd(), 'results_' + option)

    # Create the folder if it doesn't exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    plots.setResultsDir(log_directory)

    print("Option:", option, "-- results are written into the directory:", log_directory)
    logging.basicConfig(filename=os.path.join(log_directory, "log.txt"),
                        encoding='utf-8',
                        level=logging.INFO,
                        format='%(message)s')

    # train_index = int(videos * 0.5)
    # val_index = int(train_index + 50)
    # test_index = int(val_index + 50)
    # video_index = int((train_index + val_index + test_index) // 2)
    

    # Disable logging messages
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Set GPU memory growth
    def setup_gpu_memory_growth() -> None:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        print(f"{len(physical_devices)} GPU available")

    # Check if GPU is available
    def gpu():
        setup_gpu_memory_growth()
        if tf.test.is_gpu_available():
            print("TensorFlow is running on GPU")
            print("")
        else:
            print("TensorFlow is running on CPU")
            print("")

    if __name__ == '__gpu__':
        gpu()

    logPrint("")
    logPrint("{} training videos and {} epochs chosen.".format(int(videos * num_repetitions), int(epochs)))
    logPrint("")

    # Define the native and modified paths
    native_paths = [
                       ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
                        "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
                        "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]
                   ] * 4

    modified_paths = [
        ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay0.74wtPerc_2_IF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay0.74wtPerc_OF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_GA1percent_IF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_GA1percent_OF"],
        ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay0.74wtPerc_2_IF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay0.74wtPerc_OF"],
        ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"],
        ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_GA1percent_IF",
         "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_GA1percent_OF"]
    ]

    # Get the selected paths based on the chosen option
    selected_native_paths = native_paths[["MIX", "FA", "DA", "GA"].index(option)]
    logPrint("Native paths: {}".format(selected_native_paths))
    selected_modified_paths = modified_paths[["MIX", "FA", "DA", "GA"].index(option)]
    logPrint("Chem mod paths: {}".format(selected_modified_paths))

    native_videos, native_labels = get_videos(selected_native_paths, label=1, num_videos=video_index)
    modified_videos, modified_labels = get_videos(selected_modified_paths, label=0, num_videos=video_index)
    print("-=-=-"+str(len(native_videos))+",-=-=-"+str(len(native_labels))+",-=-=-"+",-=-=-"+str(len(modified_videos))+",-=-=-"+",-=-=-"+str(len(modified_labels)))
    
    

  
    
    #-------------------from--------------------#
    # Load the EfficientNetB0 model without the top layer
    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False
    # Create a sequential model
    model = Sequential([
        # Rescaling the input to the range [0, 1]
        Rescaling(scale=255),
        # TimeDistributed wrapper to apply the same transformation to each sample in the batch
        TimeDistributed(base_model),
        # Dropout layer to prevent over-fitting
        Dropout(0.2),
        # Dense layer to output the probability of each class
        Dense(10),
        # Pooling layer to reduce the dimensionality of the input
        GlobalAveragePooling3D()
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_directory, "model_fit.tsv"), separator="\t")


    times = 0 # Indicates that the 'times'th cycle is in progress
    all_history = []

    # Repeat 'num_repetitions' times using the for loop
    for i in range(num_repetitions):
        start_index = int(videos * i)
        logPrint("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+_+-+-+times: "+str(i)+", start_index: "+str(start_index))
        # Pass different indexes and label lists in each iteration
        history = fit_once(videos, epochs, option, log_directory, native_videos, native_labels, modified_videos, modified_labels, model, csv_logger, start_index)
        all_history.append(history)
    logPrint("start to call plot_accuracy_and_loss_all_history...")
    # Call the plot_accuracy_and_loss function
    plots.plot_accuracy_and_loss_all_history(all_history)
    # plots.plot_accuracy_and_loss(training_accuracy, validation_accuracy, training_loss, validation_loss)
    logPrint("plot_accuracy_and_loss_all_history finished.")
    
    # --------------------to--------------- #
   # --------------------test from--------------- #
    test_accuracy_list = []
    predictions_list = []
    test_videos_tensor_list = []
    test_vid_paths_list = []
    num_test_repetitions = 5 
    test_num = 20 # test 20 videos each time
    for i in range(num_test_repetitions):
        test_accuracy_list,predictions_list,test_videos_tensor_list,test_vid_paths_list,num_test_repetitions
        start_test_index = int(len(native_videos) - test_num*(i+1))
        test_index = int(len(native_videos)- test_num*i)
        logPrint("-+-+-+-+-+-+-+-test_times: "+str(i)+", get test from : "+str(start_test_index)+" to "+str(test_index)+"-+-+-+-+-+-+-+-")
        test_videos_tensor, test_vid_paths, test_accuracy, predictions = test_one(log_directory, native_videos, native_labels, modified_videos, modified_labels, model, start_test_index, test_index)

        test_accuracy_list.append(test_accuracy)
        predictions_list.append(predictions)
        test_videos_tensor_list.append(test_videos_tensor)
        test_vid_paths_list.append(test_vid_paths)
        

    # get accurancy
    total_accuracy = sum(test_accuracy_list)
    average_accuracy = total_accuracy / len(test_accuracy_list)
    del(test_accuracy_list)

    # Print the test accuracy
    logPrint("")
    logPrint("{} test accuracy: {:.2f}%".format(option, average_accuracy * 100))
    logPrint("")


    # Call the plot_predictions function
    # plots.plot_predictions(predictions, test_videos_tensor, test_vid_paths)
    plots.plot_multiple_predictions(predictions_list, test_videos_tensor_list, test_vid_paths_list)
    del(test_videos_tensor)
    del(predictions_list)
    del(test_videos_tensor_list)
    del(test_vid_paths_list)
    # --------------------test to--------------- #

   
    logging.shutdown()

    print("")
    print("Finished.")
    print("")

    if len(sys.argv) > 2:
        if '-q' in sys.argv[2]:
            return

    plt.show()


if __name__ == "__main__":
    main()