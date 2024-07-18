import tensorflow as tf
import os
import numpy as np
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Set the path to the dataset directory
data_dir = r'E:\soft computing\dataset\DevanagariHandwrittenCharacterDataset\Train'
print("hi")
# Define the mapping from Hindi letter to class index
# Define the mapping from Hindi letter to class index
class_map = {
    'character_1_ka': 0,
    'character_2_kha': 1,
    'character_3_ga': 2,
    'character_4_gha': 3,
    'character_5_kna': 4,
    'character_6_cha': 5, 
    'character_7_chha': 6,
    'character_8_ja': 7,
    'character_9_jha': 8,
    'character_10_yna': 9,
    'character_11_taamatar': 10,
    'character_12_thaa': 11,
    'character_13_daa': 12,
    'character_14_dhaa': 13,
    'character_15_adna': 14,
    'character_16_tabala': 15,
    'character_17_tha': 16,
    'character_18_da': 17,
    'character_19_dha': 18,
    'character_20_na': 19,
    'character_21_pa': 20,
    'character_22_pha': 21,
    'character_23_ba': 22,
    'character_24_bha': 23,
    'character_25_ma': 24,
    'character_26_yaw': 25,
    'character_27_ra': 26,
    'character_28_la': 27,
    'character_29_waw': 28,
    'character_30_motosaw': 29,
    'character_31_petchiryakha': 30,
    'character_32_patalosaw': 31,
    'character_33_ha': 32,
    'character_34_chhya': 33,
    'character_35_tra': 34,
    'character_36_gya': 35,
    'digit_0': 36,
    'digit_1': 37,
    'digit_2': 38,
    'digit_3': 39,
    'digit_4': 40,
    'digit_5': 41,
    'digit_6': 42,
    'digit_7': 43,
    'digit_8': 44,
    'digit_9': 45,
}


# Load the dataset and preprocess it for training
data = []
labels = []
for letter in os.listdir(data_dir):
    for filename in os.listdir(os.path.join(data_dir, letter)):
        img = load_img(os.path.join(data_dir, letter, filename), target_size=(32, 32))
        img_array = img_to_array(img)
        data.append(img_array)
        try:
            labels.append(class_map[letter])
        except KeyError:
            print(f"Skipping image {filename} with unknown label {letter}")
            continue
data = np.array(data)
labels = np.array(labels)
data /= 255.
print("hello2")
# Split the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the CNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Dropout(0,5),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# Train the model
model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=3)

# Save the model
model.save('new11.h5')