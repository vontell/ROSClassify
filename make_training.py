import os
from PIL import Image

root = "raw/"
saveDir = "training/"

size = 41.0*10

directories = {'l': 'lego', 'c': 'coral', 's': 'sand', 'f': 'floor', 'u': 'unknown'}

count = 786
currentFile = 19
for subdir, dirs, files in os.walk(root):
    for file in [files[currentFile]]:#files[0:2]:
        print(file)
        if file.endswith("png"):
            img = Image.open(root + file)
            img.show()
            xMod = range(int(img.width / size))
            yMod = range(int(img.height / size))
            print(len(xMod)*len(yMod))
            for x in xMod:
                for y in yMod:
                    cropped = img.crop((x*size, y*size, x*size+size, y*size+size)).resize((200,200))
                    cropped.show()
                    classification = raw_input("class: ")
                    path = saveDir + directories[classification] + "/image" + str(count) + ".png"
                    cropped.save(path, 'png')
                    print(path)
                    count += 1