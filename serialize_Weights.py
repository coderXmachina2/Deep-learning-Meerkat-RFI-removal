#script that creates weights
import pickle
from psrchive import psrchive as psrchive
import numpy as np

print("Program that computes and serializes coastguard weight into pickle file.")

#opens training dataset
with open('train_dataset.pickle', 'rb') as fid:
	trainDataset = pickle.load(fid)

with open('valid_dataset.pickle', 'rb') as fid:
	validDataset = pickle.load(fid)


#some of the training instances repeat themselves...
#gets unique instance

trains = np.array(trainDataset[2])
uniqueTrains = np.unique(trains)

vals = np.array(validDataset[2])
uniqueVals = np.unique(vals)

print("Need to compute: " + str(len(uniqueTrains)) + " Weights from training\n")
print("Need to compute: " + str(len(uniqueVals)) + " Weights from training\n")

#initiate empty dictionary
label_Weight_Dict = {}
label_np_max_Weight_Dict = {}

countTrainWeights = 1
countValWeights = 1

#computes weights and saves to dictionary
print("Now computing training weights")
for i in range(0, len(uniqueTrains)):
	labelContents = psrchive.Archive_load(str(uniqueTrains[i]))    
	weights = labelContents.get_weights() #get weights is a psrchive thing this is computationally demanding
   	label_Weight_Dict.update({ uniqueTrains[i]: weights })
    	label_np_max_Weight_Dict.update({ uniqueTrains[i]: np.max(weights)})
    	print(str(countTrainWeights) + "/" + str(len(uniqueTrains)) + " training  weights computed.")
	countTrainWeights += 1

#are you sure this doesnt work. You should have tried it twice
print("\nNow computing validation weights")
#Just append it to to the other end.
for t in range(0, len(uniqueVals)):
	labelContents = psrchive.Archive_load(str(uniqueVals[t]))    
	weights = labelContents.get_weights() #get weights is a psrchive thing this is computationally demanding
   	label_Weight_Dict.update({ uniqueVals[t]: weights })
    	label_np_max_Weight_Dict.update({ uniqueVals[t]: np.max(weights) })
    	print(str(countValWeights) + "/" + str(len(uniqueVals)) + " training  weights computed.")
	countValWeights += 1

#save to external pickle
with open('label_weight.pickle', "wb") as fid: #this creates file, to write
    pickle.dump(label_Weight_Dict, fid)
with open('label_np_max_weight.pickle', "wb") as fid: #this creates file, to write
    pickle.dump(label_np_max_Weight_Dict, fid)

print("\nPickle file created.")

