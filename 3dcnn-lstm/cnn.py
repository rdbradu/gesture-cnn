import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from dataset import extract_dataset
from process import extract_frames

class Conv3DModel(tf.keras.Model):
  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.convLSTM(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.out(x)
  
  def __init__(self):
    super(Conv3DModel, self).__init__()
    self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')
   
    self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
    self.flatten =  tf.keras.layers.Flatten(name="flatten")

    self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
    self.out = tf.keras.layers.Dense(5, activation='softmax', name="output")

def main():
    extract_dataset("./20bnjester/Train/", "./trainset/", "./20bnjester/Train.csv", "train")
    extract_dataset("./20bnjester/Validation/", "./validationset/", "./20bnjester/Validation.csv", "validation")
    train_x, train_y = extract_frames("train.csv", "./trainset/")
    val_x, val_y = extract_frames("validation.csv", "./validationset/")

    scaler = StandardScaler()
    train_x = np.array(train_x, dtype=np.float32)

    scaled_train_x  = scaler.fit_transform(train_x.reshape(-1, 15*64*64))
    scaled_train_x  = scaled_train_x.reshape(-1, 15, 64, 64, 1)

    val_x = np.array(val_x, dtype=np.float32)

    scaled_val_x  = scaler.fit_transform(val_x.reshape(-1, 15*64*64))
    scaled_val_x  = scaled_val_x.reshape(-1, 15, 64, 64, 1)

    train_dataset = tf.data.Dataset.from_tensor_slices((scaled_train_x, train_y))
    val_dataset = tf.data.Dataset.from_tensor_slices((scaled_val_x, val_y))

    model = Conv3DModel()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()


    # In[204]:


    # Loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    # Accuracy
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')


    # In[205]:


    @tf.function
    def train_step(image, targets):
        with tf.GradientTape() as tape:
            # Make a prediction on all the batch
            predictions = model(image)
            # Get the error/loss on these predictions
            loss = loss_fn(targets, predictions)
        # Compute the gradient which respect to the loss
        grads = tape.gradient(loss, model.trainable_variables)
        # Change the weights of the model
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # The metrics are accumulate over time. You don't need to average it yourself.
        train_loss(loss)
        train_accuracy(targets, predictions)


    # In[206]:


    @tf.function
    def valid_step(image, targets):
        predictions = model(image)
        t_loss = loss_fn(targets, predictions)
        # Set the metrics for the test
        valid_loss(t_loss)
        valid_accuracy(targets, predictions)


    # #### here I use the checkpoints
    # read more:
    # https://www.tensorflow.org/beta/guide/checkpoints

    # In[207]:


    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, 'training_checkpoints/tf_ckpts', max_to_keep=10)
    ckpt.restore(manager.latest_checkpoint)


    # In[ ]:


    epoch = 10
    batch_size = 32
    b = 0
    training_acc = []
    validation_acc = []
    for epoch in range(epoch):
        # Training set
        for images_batch, targets_batch in train_dataset.batch(batch_size):
            train_step(images_batch, targets_batch)
            template = '\r Batch {}/{}, Loss: {}, Accuracy: {}'
            print(template.format(
                b, len(train_y), train_loss.result(), 
                train_accuracy.result()*100
            ), end="")
            b += batch_size
        # Validation set
        for images_batch, targets_batch in val_dataset.batch(batch_size):
            valid_step(images_batch, targets_batch)

        template = '\nEpoch {}, Valid Loss: {}, Valid Accuracy: {}'
        print(template.format(
            epoch+1,
            valid_loss.result(), 
            valid_accuracy.result()*100)
        )
        training_acc.append(float(train_accuracy.result()*100))
        validation_acc.append(float(valid_accuracy.result()*100))
        ckpt.step.assign_add(1)
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        valid_loss.reset_state()
        valid_accuracy.reset_state()
        train_accuracy.reset_state()
        train_loss.reset_state()


        # In[209]:


        print(manager.checkpoints)


        # In[ ]:


        # plote Accuracy / epoch
        # plt.plot(training_acc)
        # plt.plot(validation_acc)

        # plt.ylabel('Accuracy')
        # plt.xlabel('Epochs')
        # plt.show()


        # In[217]:


        # save the model for use in the application
        model.save_weights('weights/cnn.weights.h5')


# main()