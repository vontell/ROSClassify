from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('grand_challenge_trained_4.h5')
size = 41*10

def classify_image(path_to_image):

    img = Image.open(path_to_image).convert("RGBA")
    xMod = range(int(img.width / size))
    yMod = range(int(img.height / size))
    for x in xMod:
        for y in yMod:
            cropped = img.crop((x*size, y*size, x*size+size, y*size+size)).resize((200,200))
            Rvals = np.array(cropped.getdata(band=0)).reshape((200,200))
            Gvals = np.array(cropped.getdata(band=1)).reshape((200,200))
            Bvals = np.array(cropped.getdata(band=2)).reshape((200,200))
            hues = np.array(cropped.convert("HSV").getdata(band=0)).reshape((200,200))

            # Normalize
            Rvals = Rvals / 255.0
            Gvals = Gvals / 255.0
            Bvals = Bvals / 255.0
            hues = hues / 360.0

            RGBHimage = np.stack((Rvals, Gvals, Bvals, hues), axis=-1)
            expanded = np.expand_dims(RGBHimage, axis=0)
            result = model.predict_classes(expanded)[0]

            overlay_color = (255, 0, 0, 50) if result == 0 else ((0, 255, 0, 50) if result == 1 else (0, 0, 255, 50))
            overlay = Image.new('RGBA', (size,size), overlay_color)
            img.paste(overlay, box=(x*size, y*size))
            print(result)
    img.show()

classify_image("raw/image4.png")