import tensorflow as tf

# Set up paths
train_tfrecord = '/Users/viraajkochar/SignLanguageInterpreter/data/ASL Letters/train/Letters.tfrecord'
valid_tfrecord = '/Users/viraajkochar/SignLanguageInterpreter/data/ASL Letters/valid/Letters.tfrecord'
test_tfrecord = '/Users/viraajkochar/SignLanguageInterpreter/data/ASL Letters/test/Letters.tfrecord'
label_map_path = '/Users/viraajkochar/SignLanguageInterpreter/data/ASL Letters/train/Letters_label_map.pbtxt'

# Load the label map
def load_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, 'r') as file:
        for line in file:
            if "id" in line:
                id = int(line.split(": ")[1])
            if "display_name" in line:
                name = line.split(": ")[1].strip().replace('"', '')
                label_map[id] = name
    return label_map

label_map = load_label_map(label_map_path)
print("Label Map:", label_map)

# Parse function for TFRecord
def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    label = example['image/class/label']
    image = tf.image.resize(image, (224, 224))
    return image, label

# Create datasets
train_dataset = tf.data.TFRecordDataset(train_tfrecord).map(parse_tfrecord_fn)
valid_dataset = tf.data.TFRecordDataset(valid_tfrecord).map(parse_tfrecord_fn)
test_dataset = tf.data.TFRecordDataset(test_tfrecord).map(parse_tfrecord_fn)

# Batch and shuffle datasets
batch_size = 32
train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=5
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Save the model
model.save('asl_model.h5')
