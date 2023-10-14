import tensorflow as tf

from clearml import Task
task = Task.init(project_name="test_deployment",
                 task_name="tensorflow_training")

mnist = tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test /255.0

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

print(model.summary())

model.fit(x_train,y_train,epochs=20)
model.save('epoch_20.keras')
