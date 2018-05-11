import os
from PIL import Image
#import psutil
import pickle

root = "raw/"
saveDir = "training/"

size = 41*10

directories = {'l': 'lego', 'c': 'coral', 's': 'sand', 'f': 'floor', 'u': 'unknown'}

count = 786

if os.path.isfile('count.pkl'):
    with open('count.pkl', 'rb') as f:
        count = pickle.load(f)

currentFile = 19

if os.path.isfile('currentFile.pkl'):
    with open('currentFile.pkl', 'rb') as f:
        currentFile = pickle.load(f)

print(currentFile)

for subdir, dirs, files in os.walk(root):
    for file in [files[currentFile]]:#files[0:2]:
        print(file)
        if file.endswith("png"):
            img = Image.open(root + file)
            # img.show()
            xMod = range(int(img.width / size))
            yMod = range(int(img.height / size))
            print(len(xMod)*len(yMod))
            for x in xMod:
                for y in yMod:
                    cropped = img.crop((x*size, y*size, x*size+size, y*size+size))
                    cropped = cropped.resize((200,200))
                    cropped.show()
                    classification = raw_input("class: ")
                    path = saveDir + directories[classification] + "/image" + str(count) + ".png"
                    cropped.save(path, 'png')
                    print(path)
                    count += 1
                    with open('count.pkl', 'wb') as f:
                        pickle.dump(count, f)
                    
                    # for proc in psutil.process_iter():
                    #     if proc.name() == "display":
                    #         # proc.terminate()
                    #         proc.kill()

        currentFile += 1
        with open('currentFile.pkl', 'wb') as f:
            pickle.dump(currentFile, f)