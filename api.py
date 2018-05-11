from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('grand_challenge_trained_new.h5')

classes = ["coral", "lego", "floor", "sand"]
DEFAULT_PIXEL_LOCATIONS = [(110, 390), (110, 250), (320, 250), (320, 390), (530, 250), (530,390)]

# Given a path to an image, returns the classification at some points in space
# Params:
#       path_to_image - Path to the image to classify
#       points (OPTIONAL) - list of (x,y) tuples to classify, from the bottom left corner
#                           Defaults to six points in the bottom half of the image
#       img_display (OPTIONAL) - True if pictures should be printed, defaults to False
# Returns: Tuple with (x position, y position, classification probability array)
#           Prob array is [coral, lego, floor, sand]
def classify_image(path_to_image, points=DEFAULT_PIXEL_LOCATIONS, img_display=False):

    results = []
    img = Image.open(path_to_image).convert("RGBA")
    if img_display:
        img.show()
    img = img.resize((640, 480))
    patch_size = (80,80)
    for point in points:
        x, y = point
        #y = 480 - y
        box = (int(x - patch_size[0]/2), int(y - patch_size[1]/2), int(x + patch_size[0]/2), int(y + patch_size[1]/2))

        cropped = img.crop(box).resize((200,200))
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
        probs = model.predict(expanded)[0]
        # result = -1
        # if probs[3] > 0.17:
        #     result = 3
        # else:
        #     result = model.predict_classes(expanded)[0]


        if img_display:
            
            result = model.predict_classes(expanded)[0]

            # RED IS CORAL
            # GREEN IS LEGO
            # BLUE IS FLOOR
            # YELLOW IS SAND
            color = [(255, 0, 0, 50), (0, 255, 0, 50), (0, 0, 255, 50), (255, 255, 0, 50)]

            overlay_color = color[result]
            overlay = Image.new('RGBA', patch_size, overlay_color)
            img.paste(overlay, box=(box[0], box[1]))

        x, y = getXYFromPixel(point)
        final_result = (x, y, probs)
        results.append(final_result)   

    if img_display:
        img.show()

    return results

def getXYFromPixel(pixel):
    x_pix, y_pix = pixel

    # First get the x position from the fit
    distanceFromBottom = 6.497747897*1.004078138**y_pix + 3

    # Now get the y position
    midPixel = 640 / 2
    distanceFromCenter = abs(midPixel - x_pix)
    theta = distanceFromCenter / midPixel
    x_pos = np.tan((62.2/360) * np.pi)*theta*distanceFromBottom
    if x_pix < midPixel:
        x_pos *= -1

    return distanceFromBottom, x_pos

if __name__ == '__main__':
    classifs = classify_image("new_test/imgDistance2.jpg", img_display=False) # (mix - 31) (sand - 57, 53) (none - 58) (coral - 38) (cool result - 79)
    print(classifs)
