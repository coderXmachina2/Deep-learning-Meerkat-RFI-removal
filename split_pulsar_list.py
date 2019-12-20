#split full_pulsar_list.txt into random train, validation, test sets

import random
import numpy as np
import pickle

debug = False #modify this to update pulsar training set

print("Percentage split of the data:")

val = float(2)
test = float(94)
train = float(100 - (val + test))

if debug:
  validation_split=0.35
  test_split = 0.1
else:	
  validation_split=val/100  #thats why your validation is sideways
  test_split = test/100

print("Training Data: " + str(train/100))
print("Validation Data: " + str(validation_split))  #thats why your validation is sideways
print("Testing: " + str(test_split))

# with open("full_pulsar_list.txt", "r") as fid:
with open("pulsar_list.txt", "r") as fid:
  pulsar_list = fid.read().split()

print("\nList of Pulsars:")
print(pulsar_list)
print("Number of Pulsars: " + str(len(pulsar_list)))

#split into train, validation and test sets
random.seed(42)
random.shuffle(pulsar_list)
n_pulsars = len(pulsar_list)
n_train = int(np.floor((1 - validation_split - test_split) * n_pulsars))
n_valid = int(np.floor(validation_split * n_pulsars))
n_test = n_pulsars - n_train - n_valid
train = pulsar_list[0:n_train]
valid = pulsar_list[n_train : n_train + n_valid]
test = pulsar_list[n_train + n_valid : ]

if debug:
  with open("pulsar_list_train_debug.pickle", "wb") as fid:
    pickle.dump(train, fid)

  with open("pulsar_list_valid_debug.pickle", "wb") as fid:
    pickle.dump(valid, fid)

  with open("pulsar_list_test_debug.pickle", "wb") as fid:
    pickle.dump(test, fid)
else: #not in debug mode create new train, valid, test
  print("\nExecuting Production Mode. Train, Valid, and Test lists.")
  with open("pulsar_list_train.pickle", "wb") as fid:
	print("Creating training pickle")
	pickle.dump(train, fid)

  with open("pulsar_list_valid.pickle", "wb") as fid:
        print("Creating valid  pickle")
	pickle.dump(valid, fid)

  with open("pulsar_list_test.pickle", "wb") as fid:
        print("Creating test pickle")
	pickle.dump(test, fid)

print("\nTraining, Validation, and Testing lists updated. Program Ends...")
