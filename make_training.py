import os
from PIL import Image
#import psutil
import pickle

root = "ros_images/"
saveDir = "save_ros/"

# size = 41*10
size = 80

directories = {'l': 'lego', 'c': 'coral', 's': 'sand', 'f': 'floor', 'u': 'unknown'}

count = 0

if os.path.isfile('count_ros.pkl'):
    with open('count_ros.pkl', 'rb') as f:
        count = pickle.load(f)

currentFile_ros = 0

if os.path.isfile('currentFile_ros.pkl'):
    with open('currentFile_ros.pkl', 'rb') as f:
        currentFile_ros = pickle.load(f)

print(currentFile_ros)

for subdir, dirs, files in os.walk(root):
    for file in [files[currentFile_ros]]:#files[0:2]:
        print(file)
        if file.endswith("jpg"):
            img = Image.open(root + file)
            img.show()
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
                    with open('count_ros.pkl', 'wb') as f:
                        pickle.dump(count, f)
                    
                    # for proc in psutil.process_iter():
                    #     if proc.name() == "display":
                    #         # proc.terminate()
                    #         proc.kill()

        currentFile_ros += 1
        with open('currentFile_ros.pkl', 'wb') as f:
            pickle.dump(currentFile_ros, f)