from siftUtils import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from random import random
import pickle

k = 15 # number of vocab features
n_nn = 7 # parameter for knn classifier
added_limit = 18
image_labels = ["coral","sand","floor","lego"]

## get feature labels ##
trainingGroups = [return_labels([image_labels[i]], added_limit) for i in range(len(image_labels))]
allTrainingLabels = np.concatenate(trainingGroups)
allTrainingFeatures = allTrainingLabels.reshape(-1, 128)

## cluster to build vocabulary ##
featureClassifier = KMeans(n_clusters=k, random_state=1)
featureClassifier.fit(allTrainingFeatures)

## generate term vector for each training image ##
def get_term_vector(picture):
    tv = [0 for i in range(k)]
    features = picture.reshape(-1, 128)
    for feature in features:
        result = featureClassifier.predict([feature])
        tv[int(result)] +=1
    return tv

## get term vectors and classify ##
Xtv = []
ylabels = []
for i in range(len(image_labels)):
    for picture in trainingGroups[i]:
        tv = get_term_vector(picture)
        Xtv.append(tv)
        ylabels.append(i)

tvClassifier = KNeighborsClassifier(n_neighbors=n_nn)
tvClassifier.fit(Xtv,ylabels)

out = open("vocabClassifier.pkl", "w")
pickle.dump(featureClassifier, out)
out.close()

out = open("termVectorClassifier.pkl","w")
pickle.dump(tvClassifier, out)
out.close()
