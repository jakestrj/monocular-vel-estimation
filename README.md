# speed-challenge
Work on Comma.ai speed challenge

## Architecture designs
This design was motivated by the optical flow CNN model introduced on the [FlowNet model](https://arxiv.org/pdf/1504.06852.pdf). Taking the resulting optical flow map and feeding it through several fully connected layers is used to predict speed in m/s. 

Additionally, a variable depth is introduced to encourage flow over more than two subsequent frames. We average the velocity estimates and labels over this specified depth.

Preprocessing is relatively trivial: involving cv2 gamma correction, kernel sharpening, and shapping

### Improvements
- 3d convolutions (multiple channels) as we are iterating over the time domain between subsequent frames
- [Grouped (parallel) convolutions](https://towardsdatascience.com/grouped-convolutions-convolutions-in-parallel-3b8cc847e851) are introduced before the fully connected layers to reduce training time and improve efficiency as in e.g. AlexNet.

## Validation Data
Average MSE Loss on Validation Set over 60 Epochs: ~1.1
 - Parameters: `lr`: 1e-4, `batch_size`: 16, `kernel-size`: (3x11x11), `output-size`, (, 20), `epochs`: 60

## Review and Concerns
- Part of test video around ~60sec where perpendicular cars affect ego-vehicle speed
- Train on expanded data set (KITTI) to improve test set accuracy and robustness