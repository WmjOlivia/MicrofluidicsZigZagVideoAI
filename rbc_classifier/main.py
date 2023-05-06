import getopt
import sys
import logging
import time
from video_processor import get_videos, process_dataset
import tensorflow as tf


def main(argv):
    # Set default values for parameters
    videos = 80
    epochs = 10
    batch_size = 1
    # Get parameters from command line
    try:
        opts, args = getopt.getopt(argv, "v:e:b:", ["videos=", "epochs=", "batch_size="])
    except getopt.GetoptError:
        print("main.py -v <videos> -e <epochs> -b <batch_size>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-v", "--videos"):
            val_videos = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)

    # Disable all logging messages
    logging.disable(logging.CRITICAL)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    def setup_gpu_memory_growth() -> None:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        print(f"{len(physical_devices)} GPU available")

    def main():
        setup_gpu_memory_growth()
        if tf.test.is_gpu_available():
            print("TensorFlow is running on GPU")
            print("")
        else:
            print("TensorFlow is running on CPU")
            print("")

    if __name__ == '__main__':
        main()

    print(f"{int(videos)} videos will be used for training and validation")
    print("")
    # time.sleep(1)

    # Define the healthy and ill paths
    # native_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_overfocused2ticks",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_underfocused2ticks",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
    #                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]
    #
    # modified_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/FA_0.37wtPercent",
    #                   "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
    #                   "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
    #                   "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"]

    native_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/native"]
    modified_paths = ["/home/raj/PycharmProjects/droplets_video/rbc_classifier/modified2"]

    # Select videos for training and validation sets
    native_videos, native_labels = get_videos(native_paths, label=1, num_videos=videos // 2)
    modified_videos, modified_labels = get_videos(modified_paths, label=0, num_videos=videos // 2)

    # Process data
    videos_tensor, labels_tensor = process_dataset(native_videos, modified_videos,
                                                   native_labels, modified_labels)

    # Process tensors
    dataset = tf.data.Dataset.zip((videos_tensor, labels_tensor))
    dataset.cache().shuffle(10).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    # ----------------------------------- #

    net = tf.keras.applications.EfficientNetB0(include_top=False)
    net.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=255),
        tf.keras.layers.TimeDistributed(net),
        tf.keras.layers.Dense(10),
        tf.keras.layers.GlobalAveragePooling3D()
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(dataset,
              epochs=epochs,
              callbacks=tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'))

    # # Build the model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.TimeDistributed(tf.keras.applications.ResNet50(include_top=False)),
    #     tf.keras.layers.GlobalAveragePooling3D(),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(1, activation='sigmoid'),
    # ])
    #
    # # Compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    # # Train the model
    # history = model.fit(dataset, epochs=epochs, batch_size=batch_size, verbose=1)
    # # validation_split=tf.constant(0.2)
    #
    # # Print the model summary
    # model.summary()
    # time.sleep(5)
    #
    # print("")
    # accuracy = history.history['accuracy'][-1] * 100
    # print('Accuracy: {:.2f}%'.format(accuracy))


if __name__ == "__main__":
    main(sys.argv[1:])
