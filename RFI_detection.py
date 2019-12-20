#!/usr/bin/env python

#try training on the small subset that I have downloaded from ozstar
from __future__ import absolute_import, division, print_function, unicode_literals #trying to check GPU
import os
import pickle
import random
import gc
import numpy as np

import tensorflow as tf
from tensorflow import keras
# from sklearn.utils import class_weight #TODO: use this to calculate class weights
from psrchive import psrchive as psrchive
import clfd
from load_model import load_model
from copy_data_to_local_drive import copy_data_to_local_drive
from tensorflow.python.client import device_lib

tensorboard_callback = keras.callbacks.TensorBoard(log_dir='/tmp/RFI_detection/', update_freq='batch')#, histogram_freq=1)

n_epochs = 3
batch_size = 2 #uses 17.8GB of CPU RAM, and the computer only has 32GB available, so can't increase batch size. 
               #TODO: Most RAM is probably allocated to loading coast_guard results. Consider saving them separately?

learning_rate = 0.001
debug = False
class_weights = [0.5910, 3.2486]
n_chan = 1024 #most are 928 but some are 1024
n_phase = 1024
n_pol = 4

pulsar_files_prefix = os.path.join(os.path.expanduser('~'), 'myPSRData') #this is the symbolic link. It is broken
#pulsar_files_prefix = '/media/ext_ML_datasets/pulsars/'
#but why does it keep on going to go 
pulsar_files_prefix_remote = os.path.join(os.path.expanduser('~'), 'ozstar_mountpoint')
raw_data_prefix = os.path.join(pulsar_files_prefix,'timing')
raw_data_prefix_remote = os.path.join(pulsar_files_prefix_remote,'timing')
label_prefix   = os.path.join(pulsar_files_prefix,'timing_processed')
label_prefix_remote = os.path.join(pulsar_files_prefix_remote,'timing_processed')
pulsar_files_suffix = ''

#starting from a list of folders where raw pulsar data are kept,
#create a list of files for raw pulsar data and corresponding data that has been processed by coast_guard
def list_pulsar_files(pulsar_ids):
  raw_data_files = []
  raw_data_ix = []
  label_files = []
  
  for ix, fo in enumerate(pulsar_ids):
    print('Listing ' + str(ix) + ' of ' + str(len(pulsar_ids)),'. ID=' + fo)

    if os.path.isdir(os.path.join(label_prefix, fo)):
      #NOTE: partially downloaded folders will only train on the raw_data files that have already been downloaded
      # BUT: they will trigger the rest of the files to download via rsync for the next training run
      # AND: Folders that are only remote (not local) will be included in the training list and downloaded as required
      where_to_look_for_label_files = label_prefix
      where_to_look_for_raw_data_files = raw_data_prefix
    else:
      if not os.path.exists(os.path.join(label_prefix_remote)):
        print('neither local dir: ' + os.path.join(label_prefix, fo) + ' nor remote dir: ' + os.path.join(label_prefix, fo) + ' found.')
        if debug:
          #only proceed if remote is connected
          raw_input('Check paths and press enter when ready to continue.')
        else:
          print('skipping ' + fo)
          continue
      where_to_look_for_label_files = label_prefix_remote
      where_to_look_for_raw_data_files = raw_data_prefix_remote

    #each folder of raw_data_files has exactly one label_file
    #the label_file includes cleaned data corresponding to all raw_data_files
    for root, _, files in os.walk(os.path.join(where_to_look_for_label_files, fo)):
      if os.path.basename(root) == 'cleaned':
        if len(files) == 0:
          continue
        else:
          for ff in files:
            if ff.endswith('.ar'):
              intermediate_path = root[len(where_to_look_for_label_files) + 1 :]
              intermediate_path = intermediate_path[:-len('cleaned')]
              for _, _, files2 in os.walk( os.path.join(where_to_look_for_raw_data_files, intermediate_path)):
                ix2 = 0
                for ff2 in files2:
                  if ff2.endswith('.ar'):
                    raw_data_files.append(copy_data_to_local_drive.windowsify(os.path.join(raw_data_prefix, intermediate_path, ff2)))
                    raw_data_ix.append(ix2)
                    ix2 += 1
                    label_files.append(copy_data_to_local_drive.windowsify(os.path.join(label_prefix, intermediate_path, 'cleaned', ff)))
    
  return raw_data_files, raw_data_ix, label_files

#helper function for load_raw_data_and_labels()
def tf_load_raw_data_and_labels(raw_data_files, ar_file_ix, label_files):
  [raw_data, labels] = tf.py_function(load_raw_data_and_labels, [raw_data_files, ar_file_ix, label_files], [tf.float32, tf.float32])

  raw_data.set_shape((n_chan, n_phase, n_pol))
  labels.set_shape((n_chan))
  return raw_data, labels

#given lists of files for raw pulsar data and the corresponding files with cleaned data
#load that data into a data cube ("image") and labels corresponding to which channels have been "zapped"

#this is where the error is.
def load_raw_data_and_labels(raw_data_file, ar_file_ix, label_file):
  
  print("\n-----------------------------------------------------------------------")
  #these are what come in:
  print("Data Input:")
  print(raw_data_file)
  print(ar_file_ix)
  print(label_file)
  print("-----------------------------------------------------------------------")

  raw_data_file = raw_data_file.numpy()
  label_file = label_file.numpy()
  
  #copy (rsync) files from remote storage to local storage if necessary
  if os.path.exists(os.path.join(label_prefix_remote)):
    if not (os.path.exists(label_file) and os.path.exists(raw_data_file)):
      pulsar_id = raw_data_file[len(raw_data_prefix):].strip('/')
      pulsar_id = pulsar_id[:pulsar_id.index(os.sep)] if os.sep in pulsar_id else pulsar_id
      print('calling copy_data_to_local_drive')
      copy_data_to_local_drive.copy_pulsar_file(pulsar_id)
      print('finished copy_data_to_local_drive')
  #AH_I disabled the randomizer. Reenable to mix data
  #chose either coast guard or clfd
  #if random.random() == True: #coastguard #Chris: TODO try (setting True) using just coastguard labels to test memory
  print("\nCoastguard called")   
    #print("Data: " + str(raw_data_file))
    #print("Label: " + str(label_file))

    #if coastguard data does not exist
  if not os.path.exists(label_file):
      raw_input('labe file not found: ' + label_file + '. Check ssh status and press enter when ready to continue')
    
    #original calcualtion
    #label_file_contents = psrchive.Archive_load(label_file)
    #weights = label_file_contents.get_weights() #write as py object Computationally expensive. Work with serialized weights

  with open('label_weight.pickle', "rb") as fid: #this creates file, to write
    read_Weight_File = pickle.load(fid)

  with open('label_np_max_weight.pickle', "rb") as fid: #this creates file, to write
    read_max_weight_File = pickle.load(fid)

    #original calculation
    #max_weight = np.max(weights) #This literally a single number. The biggest  number of the 2d array.

  if read_max_weight_File.get(label_file) == 0:
      labels = np.zeros(n_chan,)
  elif read_max_weight_File.get(label_file) < 0:
      print('Warning: maximum weight of ', read_max_weight_File.get(label_file), ' detected for ',label_file,'. Should be >0') #incorrect label. If this is thrown then you have quite the catastrophic failure
  else:
      labels = np.squeeze(np.round( read_Weight_File.get(label_file)[ar_file_ix,:] / ( read_max_weight_File.get(label_file) / 2) ) / 2)  #you still have to do the computation
        #original labels calculation
        #labels = np.squeeze(np.round( weights[ar_file_ix,:] / (np.max(weights) / 2) ) / 2)  #you still have to do the computation

  """
  else: #clfd
    print("\nclfd called")
    clfd_file, _ = os.path.splitext(raw_data_file)
    clfd_file += '_clfd_weights.npy'                #append the weights

    if os.path.exists(clfd_file):
      labels = np.load(clfd_file)
    else:
      #calculate, and save results for next time
      cube = clfd.DataCube.from_psrchive(raw_data_file)
      my_features = clfd.featurize(cube, features=('std', 'ptp', 'lfamp'))
      _, labels = clfd.profile_mask(my_features, q=2.0)#, zap_channels=range(150))

      labels = labels*1 #convert to 0 and 1
      labels = labels.T #transpose
      labels = -1 * (labels - 1) #invert

      np.save(clfd_file, labels)
      
      #clear large data blocks from memory
      del cube
      del my_features

  if not os.path.exists(raw_data_file):
      raw_input('raw data file not found: ', raw_data_file, '. Check ssh status and press enter when ready to continue') #but it doesnt do anything...
  """
  #print("psrchive Call!")
  raw_file_contents = psrchive.Archive_load(raw_data_file)
  raw_data = raw_file_contents.get_data()
  raw_data = np.transpose(np.squeeze(raw_data[0,:,:,:]), (1,2,0)) #convert to HxWxC

  #print(raw_data) #the raw data is a tensor of rank 1 or 2?
  #print("\nCLFD called")

  #print("Data: " + str(raw_data_file)) #string
  #print("Label: " + str(label_file)) #string
  #clear large data blocks from memory
  del raw_file_contents

  #convert raw_data to floats between 0 and 1
  my_min = np.min(raw_data)
  raw_data -= my_min
  my_max = np.max(raw_data)
  if my_max != 0:
      raw_data = np.float32(raw_data) / my_max

  raw_data = tf.image.pad_to_bounding_box(raw_data, offset_height=0, offset_width=0, target_height=n_chan, target_width=n_phase)
  padding = tf.zeros([n_chan - labels.shape[0],])
  labels = tf.concat((np.squeeze(labels), padding), axis=0)

  #Garbage collection
  #print('explicitly calling garbage collection to try and free up memory')
  gc.collect() #enforce garbage collection to free up memory, reenable garbage collection

  #print(raw_data)
  #Finish Loading, now return
  #print('Finished loading data')
  print("-----------------------------------------------------------------------\n")
  return raw_data, labels

#helper function for augment_data()
def tf_augment_data(image, label):
  im_shape = image.shape
  l_shape = label.shape
  [image, label] = tf.py_function(augment_data, [image, label], [tf.float32, tf.float32])
  image.set_shape(im_shape)
  label.set_shape(l_shape)
  return image, label

#augment data by:
# - randomly modifying contrast
# - adding white noise to all pixels
# - horizontal flip (frequency and polarisation axis unchanged)
# - translate in the phase direction (frequency and polarisation axis unchanged)
def augment_data(image, label):
  #print('augmenting data')
  #random_brightness changes all pixels equally. It doesn't seem useful if all pixels are later scaled to be in the range 0..1
  # image = tf.image.random_brightness(image, 0.25) #convert to HSV, shift V for all pixel values up or down by a random value x: -0.25<x<0.25, convert back. Pixel values assumed to be a float x: 0<x<1

  #TODO: avoid warning about n_pol=4
  # for pol_ix in range(image.shape[-1]):
  image = tf.image.random_contrast(image, 0.9, 1.1) #pixel values multiplied by x: 0.9<x<1.1
  #Is random_contrast effectively the same as increasing/decreasing the pulsar brightness without changing the background?
  # Yes, after rescaling so that all pixels are in the range 0..1
  
  #add white noise to all pixels
  if random.random() < 0.5:
    white_noise_amplitude = 0.01
    image += tf.random.normal(shape=tf.shape(image)) * white_noise_amplitude

  #make sure that values are still constrained to 0<x<1
  # my_min = np.min(image)
  # image -= my_min
  # my_max = np.max(image)
  # image = np.float32(image) / my_max
  image -= tf.reduce_min(image)
  image_max = tf.reduce_max(image)
  if image_max != 0:
    image = image / image_max

  #horizontal flip
  if random.random() < 0.5:
    image = tf.image.flip_left_right(image)

  # #vertical flipping could introduce non-physical effects, so don't do it
  # if random.random() < 0.5:
  #   image = tf.image.flip_up_down(image)
  #   my_label[0:n_chan] = np.flipud( my_label[0:n_chan] )
  
  # #keep fixed n_chan for now
  # add d_height if the routine should run on data from different telescopes with different n_chan
  #      if I use this, then modify label too
  # d_height = randrange(-1*n_chan/2, n_chan) #n_chan can halve or double
  # target_height = n_chan + d_height
  # image = tf.image.resize_image_with_crop_or_pad(image, target_height, n_phase)

  #translate image along the phase direction, with wrapping
  shift = random.randrange(-n_phase/2, n_phase/2)
  image = tf.roll(image, shift, axis=1)

  #TODO: add synthetic RF noise

  return image, label


#This is where main program starts!
if os.path.exists('train_dataset.pickle'): #what is the case for os path?
  print("\nOS path train_dataset.pickle exists!")
  with open("train_dataset.pickle", "rb") as fid:
    x_train, train_ix, y_train = pickle.load(fid)
  with open("valid_dataset.pickle", "rb") as fid:
    x_valid, valid_ix, y_valid = pickle.load(fid)
  n_train = len(train_ix) #62956
  n_valid = len(valid_ix) #7252
else:
  #load lists of pulsars
  print("\nTrain_dataset and Valid_dataset not detected. Debug Pathway Enabled")
  if debug:
    with open("pulsar_list_train_debug.pickle", "rb") as fid: #fine
      train = pickle.load(fid)

    with open("pulsar_list_valid_debug.pickle", "rb") as fid: #fine
      valid = pickle.load(fid)
  else:
    with open("pulsar_list_train.pickle", "rb") as fid: #fine
      train = pickle.load(fid)

    with open("pulsar_list_valid.pickle", "rb") as fid:
      valid = pickle.load(fid)

  if debug:
    n_train = 2
    n_valid = 2
    n_test  = 1
    train = train[0:n_train]
    valid = valid[0:n_valid]

  train_raw_data_files, train_ix, train_label_files = list_pulsar_files(train)
  valid_raw_data_files, valid_ix, valid_label_files = list_pulsar_files(valid)

  n_train = len(train_raw_data_files)
  n_valid = len(valid_raw_data_files)

  if debug:
    n_train = min(n_train, 2)
    n_valid = min(n_valid, 2)
    train_raw_data_files = train_raw_data_files[0:n_train]
    valid_raw_data_files = valid_raw_data_files[0:n_valid]

    train_ix = train_ix[0:n_train]
    valid_ix = valid_ix[0:n_valid]

    train_label_files = train_label_files[0:n_train]
    valid_label_files = valid_label_files[0:n_valid]

  #create tf.data.Dataset
  #create training set
  x_train = train_raw_data_files
  y_train = train_label_files
  #create validation set
  x_valid = valid_raw_data_files
  y_valid = valid_label_files

  with open('train_dataset.pickle', "wb") as fid: #this creates file, to write
    pickle.dump([x_train, train_ix, y_train], fid)
    #write function
  with open('valid_dataset.pickle', "wb") as fid:
    pickle.dump([x_valid, valid_ix, y_valid], fid)
    #write function

print("\nMain Program Starts")

print("\n")
print(pulsar_files_prefix)
print(pulsar_files_prefix_remote)
print(raw_data_prefix)
print(raw_data_prefix_remote)
print(label_prefix )
print(label_prefix_remote)
print("\n")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("List local devices:" + str(device_lib.list_local_devices()) + "\n\n") 
# #test/debug map functions
# a = []
# b = []
# for ix in range(len(x_train)):
#     tmp1, tmp2 = load_raw_data_and_labels(x_train[ix], train_ix[ix], y_train[ix])
#     tmp1, tmp2 = augment_data(tmp1, tmp2)
#     a.append(tmp1)
#     b.append(tmp2)
# x_train = a
# y_train = b

print('Combining ' + str(len(train_ix)) + ' samples into the training dataset...')
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, train_ix, y_train))

print('Combining ' + str(len(valid_ix)) + ' samples into the validation dataset...\n') #shouldnt this be validation?
valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, valid_ix, y_valid))

train_dataset = train_dataset.shuffle(n_train)#.repeat()
train_dataset = train_dataset.map(tf_load_raw_data_and_labels) #
train_dataset = train_dataset.map(tf_augment_data)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(1) #>1 leads to multiple threads, which orphan the rsync processes and then hang waiting for zombies to finish

valid_dataset = valid_dataset.map(tf_load_raw_data_and_labels) #
valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)

print('\nData are now ready to use')

# Available Neural models
# naive_resnet
# custom_resnet
# keras_resnet
# NASNet

modelType = 'keras_resnet' 

print('\nLoading '  + modelType  + ' model...\n')

model = load_model(modelType, n_chan, n_phase, n_pol)

print('\nPrinting out a summary to the terminal')
model.summary()

# print('saving the graph as a PNG') #routine sometimes hangs here.
keras.utils.plot_model(model, 
			'RFI_detection_model.png', 
			show_shapes=True)

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
						patience=10, 
						verbose=1, 
						mode='min')

mcp_save = keras.callbacks.ModelCheckpoint('RFI_detection_model-{epoch:02d}-{val_loss:.2f}.hdf5', 
						save_best_only=False, 
						monitor='val_loss', 
						mode='min')

reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
							factor=0.1, 
							patience=7, 
							verbose=1, 
							epsilon=1e-4, 
							mode='min')

print('\nCompiling the model\n')
model.compile(loss='binary_crossentropy', 
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
              metrics=['accuracy'])

#keras training
#TODO: try using a generator for data augmentation
print('\nModel compiled. Start training!\n\n')

#currently throwing a wobbly here
model.fit(train_dataset,
          class_weight=class_weights,
          epochs=n_epochs,
          validation_data=valid_dataset, 
          validation_freq=1,             
          callbacks=[
                    earlyStopping,
                    mcp_save,
                    reduce_lr_loss,
                    tensorboard_callback 
                    ])

modelName = 'trained_' + modelType + "_" + str(n_epochs) + "ep" + '.h5' #get this to update each time you have run it.

print('\nModel training complete! Now saving model as: ' + str(modelName))

model.save(modelName) 

print('\nModel saved. Onwards to evaluation stage.')

# #training without keras
# train_log_dir = '/tmp/logs/RFI_detection/train'
# valid_log_dir = '/tmp/logs/RFI_detection/valid'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

# valid_loss = tf.keras.metrics.Mean(name='validation_loss')
# valid_accuracy = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')

# @tf.function
# def train_step(images, labels):
#   with tf.GradientTape() as tape:
#     # keras.backend.set_learning_phase(1) #another option for setting training mode on (if activated, add an equivalent part to the valid_step)
#     predictions = model(images, training=True)
#     # unweighted_loss = loss_object(labels, predictions) #DONE: try weighted loss since there are more unnoisy channels than noisy, and very few noisy phases
#     # # loss = unweighted_loss * class_weights #This is not how you do it
#     # loss = unweighted_loss
#     loss = tf.nn.weighted_cross_entropy_with_logits(labels=labels,
#                                                     logits=predictions,
#                                                     pos_weight = class_weights[0] / class_weights[1])
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#   train_loss(loss)
#   train_accuracy.update_state(labels, predictions)

# @tf.function
# def valid_step(images, labels):
#   predictions = model(images, training=False)
#   v_loss = tf.nn.weighted_cross_entropy_with_logits(labels=labels,
#                                                     logits=predictions,
#                                                     pos_weight = class_weights[0] / class_weights[1])

#   valid_loss(v_loss)
#   valid_accuracy.update_state(labels, predictions)


# # Reset the metrics for the next epoch
# train_loss.reset_states()
# train_accuracy.reset_states()
# valid_loss.reset_states()
# valid_accuracy.reset_states()
# batch_counter = 0
# print('start training...')
# best_valid_loss_so_far = np.inf
# for epoch in range(n_epochs):
#   starting_epoch = True
#   for images, labels in train_dataset:
#     # tf.random.set_seed(1) #try (and fail) to remedy memory leak as per https://github.com/tensorflow/tensorflow/issues/19671
#     train_step(images, labels)

#     with train_summary_writer.as_default():
#       tf.summary.scalar('loss', train_loss.result(), step=batch_counter)
#       tf.summary.scalar('accuracy', train_accuracy.result(), step=batch_counter)
    
#     #break out of for loop at end of epoch (dataset is endless because of repeat())
#     batch_counter += 1
#     if ((batch_counter + 1) * batch_size) % n_train <= batch_size:
#       break

#   for valid_images, valid_labels in valid_dataset:
#     valid_step(valid_images, valid_labels)
#   with valid_summary_writer.as_default():
#     tf.summary.scalar('loss', valid_loss.result(), step=batch_counter)
#     tf.summary.scalar('accuracy', valid_accuracy.result(), step=batch_counter)

#   template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
#   print(template.format(epoch+1,
#                         train_loss.result(),
#                         train_accuracy.result()*100,
#                         valid_loss.result(),
#                         valid_accuracy.result()*100))

#   if valid_loss.result() < best_valid_loss_so_far:
#     best_valid_loss_so_far = valid_loss.result()
#     fname = os.path.join(model_savedir, my_date + model_fname + str(epoch) + '.h5')
#     print('best model so far. Saving to: ' , fname)
#     model.save(fname)

#   # Reset the metrics for the next epoch
#   train_loss.reset_states()
#   train_accuracy.reset_states()
#   valid_loss.reset_states()
#   valid_accuracy.reset_states()
# print('training finished')

#TODO: test in a separate script
# model.evaluate(test_dataset)
