import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to load and preprocess images
def load_and_preprocess(data_dir, img_height=224, img_width=224, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# Function to create MobileNetV2 model
def create_mobilenet_model(img_height=224, img_width=224, num_classes=4):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Function to apply K-means clustering
def apply_kmeans(X_train_features, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_train_features)
    return kmeans

# Function to train and evaluate Naive Bayes classifier
def train_and_evaluate_naive_bayes(X_train_features, y_train, X_test_features, y_test):
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train_features, y_train)
    nb_pred = naive_bayes.predict(X_test_features)

    return nb_pred

# Function to train and evaluate SVM classifier
def train_and_evaluate_svm(X_train_features, y_train, X_test_features, y_test):
    svm_classifier = SVC(random_state=42)
    svm_classifier.fit(X_train_features, y_train)
    svm_pred = svm_classifier.predict(X_test_features)

    return svm_pred

# Function to encode labels
def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels

# Main function
def main():
    data_dir = 'path/to/CelestialObjects'
    img_height, img_width = 224, 224
    batch_size = 32

    # Load and preprocess images
    train_generator, validation_generator = load_and_preprocess(data_dir, img_height, img_width, batch_size)

    # Create MobileNetV2 model
    mobilenet_model = create_mobilenet_model(img_height, img_width, num_classes=len(train_generator.class_indices))

    # Train MobileNetV2 model
    mobilenet_model.fit(train_generator, epochs=10, validation_data=validation_generator)

    # Extract features using MobileNetV2
    X_train_features = mobilenet_model.predict(train_generator)
    X_test_features = mobilenet_model.predict(validation_generator)

    # Apply K-means clustering
    kmeans = apply_kmeans(X_train_features)

    # Use K-means labels as pseudo-labels for training classifiers
    pseudo_labels = kmeans.predict(X_train_features)

    # Train and evaluate Naive Bayes
    nb_pred = train_and_evaluate_naive_bayes(X_train_features, pseudo_labels, X_test_features, encode_labels(validation_generator.classes))

    # Train and evaluate SVM
    svm_pred = train_and_evaluate_svm(X_train_features, pseudo_labels, X_test_features, encode_labels(validation_generator.classes))

    # Evaluate MobileNetV2 model
    test_loss, test_acc = mobilenet_model.evaluate(validation_generator)
    print(f'MobileNetV2 Test Accuracy: {test_acc:.2f}')

    # Evaluate Naive Bayes model
    evaluate_model('Naive Bayes', validation_generator.classes, nb_pred)

    # Evaluate SVM model
    evaluate_model('SVM', validation_generator.classes, svm_pred)

# Run the main function
if __name__ == "__main__":
    main()