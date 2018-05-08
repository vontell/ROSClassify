# ROSClassify

# Instructions
1. Download the [raw images](https://drive.google.com/open?id=1cjuvRTpggDX2W_G4E-m1HGAAvhdamkRq) and [training images](https://drive.google.com/open?id=1CD3ccvi3KJQEOYaqbz8TJHP4_NYZ3N1g) from Google Drive
2. Extract these folders to the root folder of the project (i.e. you should now have two new folders, `raw` and `training`)
3. Run `python make_training.py` to load image segments and classify (please increment the image pointer and file count appropriately, or old image segments will be overwritten) PLEASE TALK TO AARON BEFORE DOING THIS
4. Run `python train.py` to train the model on lego vs coral vs floor. Note that this only uses 91 images from each class, so you can change.
5. Run `python3 api.py` to classify segments from an image