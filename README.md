# PROBA-V Super Resolution Algorithm

Notes and Code for PROBA-V Super Resolution Algorithm

## Objective

Given multiple images of each of 78 Earth locations, develop an algorithm to fuse them together into a single one.
The result will be a "super-resolved" image that is checked against a high resolution image taken from the same satellite.

## Data

The data is composed of radiometrically and geometrically corrected Top-Of-Atmosphere (TOA) reflectances for the RED and NIR spectral bands at 300m and 100m resolution in Plate Carrée projection. The 300m resolution data is delivered as 128x128 grey-scale pixel images, the 100m resolution data as 384x384 grey-scale pixel images. The bit-depth of the images is 14, but they are saved in a 16-bit .png-format (which makes them look relatively dark if opened in typical image viewers).

- A quality map indicates which pixels in the image are concealed (i.e. clouds, cloud shadows, ice, water, missing, etc) and which should be considered clear.

- In total, the dataset contains 1450 scenes, which are split into 1160 scenes for training and 290 scenes for testing. On average, each scene comes with 19 different low resolution images and always with at least 9.

- Submit a 384x384 image for each of the 290 test-scenes, for which we will not provide a high resolution image.

## Todo

- [X] Datareader
- [ ] Dummy Model
- [ ] Base Model
- [ ] Objective function
- [ ] Model Evaluation

## Links

- [Dataset](https://kelvins.esa.int/proba-v-super-resolution/data/)
- [Scoring](https://kelvins.esa.int/proba-v-super-resolution/scoring/)
- [Submission Rules](https://kelvins.esa.int/proba-v-super-resolution/submission-rules/)
- [Post mortem](https://kelvins.esa.int/proba-v-super-resolution-post-mortem/leaderboard/)
