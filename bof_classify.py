from siftUtils import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import pickle


inp = open("vocabClassifier.pkl", "r")
featureClassifier = pickle.load(inp)
inp.close()

inp = open("termVectorClassifier.pkl", "r")
tvClassifier = pickle.load(inp)
inp.close()

k = featureClassifier.n_clusters # number of vocab features
added_limit = 18
image_labels = ["coral","sand","floor","lego"]

#TODO
#### slice image ##
##    xMod = range(int(img.width / size))
##    yMod = range(int(img.height / size))
##    for x in xMod:
##        for y in yMod:
##            cropped = img.crop((x*size, y*size, x*size+size, y*size+size)).resize((200,200))
##            Rvals = np.array(cropped.getdata(band=0)).reshape((200,200))
##            Gvals = np.array(cropped.getdata(band=1)).reshape((200,200))
##            Bvals = np.array(cropped.getdata(band=2)).reshape((200,200))
##            hues = np.array(cropped.convert("HSV").getdata(band=0)).reshape((200,200))
##
##            # Normalize
##            Rvals = Rvals / 255.0
##            Gvals = Gvals / 255.0
##            Bvals = Bvals / 255.0
##            hues = hues / 360.0
##
##            RGBHimage = np.stack((Rvals, Gvals, Bvals, hues), axis=-1)
#### save the 48 split picture as png into a directory called "ToClassify"
##
##

pictureLabels = return_labels(["coral"], added_limit) ## change directory to "ToClassify" once above is done
allTrainingFeatures = pictureLabels.reshape(-1, 128)

## generate term vector for each training image ##
def get_term_vector(picture):
    tv = [0 for i in range(k)]
    features = picture.reshape(-1, 128)
    for feature in features:
        result = featureClassifier.predict([feature])
        tv[int(result)] +=1
    return tv

## get term vectors and classify ##
classify = lambda p: image_labels[int(tvClassifier.predict([get_term_vector(p)]))]
classifications = map(classify, pictureLabels)
print classifications 
