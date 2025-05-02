# Kaggle : Image Matching Challenge 2025

## Overview ([URL](https://www.kaggle.com/competitions/image-matching-challenge-2025))

You'll develop machine learning algorithms that can figure out which images belong together and use them to reconstruct accurate 3D scenes. This innovation advances the field of computer vision and enables new applications in augmented reality, robotics, and AI.

## Description

Note: This is the third Image Matching Challenge. It builds on 2024's Image Matching Challenge. This year, we’re taking it a step further and challenging you to determine how images should be grouped together or discarded, in addition to reconstructing 3D scenes.

Imagine sitting down to solve a jigsaw puzzle. But when you open the box, you discover that your pieces have been jumbled together with more pieces from other puzzle sets! How do you determine which pieces are for your puzzle and which belong to other sets?

Reconstructing a 3D scene from a set of possibly related images is a core problem in computer vision. While current methods work well in controlled environments with professional equipment, they struggle with real-world image collections.

Online image collections are messy and often contain a mix of unrelated photos or visually similar images that confuse reconstruction models. For example, two nearly identical sides of a monument might be mistaken for the same view, or completely unrelated images (like a photo of a latte taken near a landmark) could accidentally be grouped together. Existing methods use GPS data or video sequences to help, but these aren't always available or reliable, making large-scale applications unreliable.

This competition challenges you to identify which images should be grouped and which should be discarded in 3D scene reconstruction. This will improve Structure from Motion (SfM) techniques and help generate more accurate 3D models from diverse image collections.

Your work could make crowdsourced images more useful for large-scale reconstructions, benefiting areas like urban planning and scientific research.

## Evaluation

The ground-truth data consists of multiple datasets `D_k`. Each contains one or more scenes `S_ki` with images `I_kiz`. The different scenes belonging to one dataset do not overlap: they show different regions or objects, but are similar in appearance, like two sides of the same building, or two different trees. Each dataset may have a different number of scenes and images. Each dataset may or may not contain "outlier" images, which do not correspond to any scene.

For training data, we provide the camera pose of each image in terms of its rotation matrix R
 and translation vector T
. Images that do not correspond to any scene are placed in an outliers folder, for which no pose is provided. For test data, we combine all images into a single folder.

The task given to participants is (1) to partition the images `I_kiz` of each dataset `D_k` into a set of clusters `C_kj` or outliers, thus restoring the original scene assignment; and (2) to reconstruct each scene independently, providing the camera pose of each image, expressed as a rotation matrix and a translation vector.

The submission score is obtained as the combination of the mean Average Accuracy (mAA) of the registered camera centers, and the clustering score. The mAA of a given cluster `C` with respect to the scene `S` is computed exactly, using the same metric as in the 2024 competition. It is equal to the ratio of the registered images of the scene found in the cluster divided by the cardinality of the scene. The clustering score is given by number images of the cluster effectively belonging to the scene, divided by the cardinality of the cluster, i.e. `|S∩C||C|`
.

The score computation involves two steps. In the first step, each scene `S_ki∈D_k` is greedily associated to the cluster `C_kij`, among those provided by the user, that maximizes the mean Average Accuracy (mAA). If other user clusters provide the same mAA, the one with higher clustering score is chosen. The cluster index associated to scene `i` in this way is denoted by `j_i`. Notice that the same user cluster can be associated to more scenes, and that all the images labeled by the user as outliers are excluded from the greedy assignment.

In the second step, the overall mAA score and clustering score of a dataset are computed by aggregating the individual values. For the clustering score this is equal to
$$
∑_i|S_{ki}∩C_{kj_i} /|∑_i|C_{kj_i}|
$$

And analogously for the mAA score.

The combined score `S_k` is the harmonic mean of the mAA and clustering scores. In this formulation, the mAA score is roughly equivalent to recall, and the clustering score to precision, and the final score is thus analogous to the standard F1 score. Finally, we average the results over the different datasets to obtain a single score.

## Submission file

For each image ID in the test set, you must predict a scene assignment and a pose. The file should contain a header and have the following format:

```bash
dataset,scene,image,rotation_matrix,translation_vector
dataset1,cluster1,image1.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster1,image2.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster2,image3.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,cluster2,image4.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
dataset1,outliers,image5.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
dataset2,cluster1,image1.png,0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9,0.1;0.2;0.3
```

The scene labels are assigned by you. They are only used to specify which images belong together, so they can contain arbitrary strings: we recommend using something simple, like `cluster1`, `cluster2`, etc. The exception is an outliers label, which should be assigned to images that cannot be registered to any other image.

The example above contains dataset1 with five images. The solution indicates that images 1 and 2 belong together, as do images 3 and 4. Image 5 does not, so it is assigned to outliers. Images that you think are outliers are not registered to other images, so the rotation matrix and translation vector are not used: we make this clear using nan for every value.

The `rotation_matrix` (a 3x3 matrix) and `translation_vector` (a 3-D vector) are written as `;`-separated vectors. Matrices are flattened into vectors in row-major order. Note that this metric does not require camera intrinsics, i.e., the calibration matrix `K`) that is usually estimated along with `R` and `T` during the 3D reconstruction process.

Note that you can group images together if you think they belong to the same scene, even if you cannot register them. You can indicate this by using the right scene label and nan values for rotation_matrix and translation_vector. For example:

```bash
dataset2,cluster1,image2.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
```

Notice that we can re-use scene labels (such as `cluster1`) for different datasets.

## Usage

```bash
chmod +x setup_dataset.sh
chmod +x train.sh

# Build dataset
./setup_dataset.sh

# Train the model
./train.sh --data_dir ./dataset/train
```