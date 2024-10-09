<h2>Tensorflow-Tiled-Image-Segmentation-Follicular-Cell (2024/10/10)</h2>

This is the first experiment for Follicular-Cell Segmentation based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1ISpL9l-Dv8k93BrqGb0V0vaab9TrQP5U/view?usp=sharing">
Tiled-Follicular-Cell-ImageMask-Dataset-V2.zip</a>, which was derived by us from 
<a href="https://drive.google.com/file/d/1t1W7tpKscqLxPApqH3JSP_zsljLHWKxQ/view?usp=sharing">IM_PatchDataset.zip.</a>.
<br>
<br>
On detail of the Tiled ImageMask Dataset, please refer to  
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Follicular-Cell">Tiled-ImageMask-Dataset-Follicular-Cell</a>
<br>
<br>
<b>Experiment Strategies</b><br>
In this experiment, we employed the following strategies.<br>

<b>1. Tiled ImageMask Dataset</b><br>
 We trained and validated a TensorFlow UNet model using the Tiled-Follicular-Cell-ImageMask-Dataset, which was tiledly-splitted to 512x512 pixels
 and reduced to 512x512 pixels image and mask dataset.<br><br>
<b>2. Tiled Image Segmentation</b><br>
We applied the Tiled-Image Segmentation inference method to predict the follicular cell regions for the mini_test images 
with a resolution of 1024x1024 pixels. 
<br><br>

<b>3. Color Space Conversion</b><br>
We applied an RGB to HSV color space conversion to the input images and test images as an image preprocessor.  
<br>
<table>
<tr>
<th>RGB</th>
<th>HSV</th>
<th>Mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/images/13_23552-18432.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/hsv_images/13_23552-18432.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/masks/13_23552-18432.png" width="320" height="auto"></td>

</tr>

</table>
<br>

<hr>
<b>Actual Tiled Image Segmentation for Images of 1024x1024 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled-inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/images/13_23552-18432.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/masks/13_23552-18432.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test_output_tiled/13_23552-18432.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/images/13_24576-8192.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/masks/13_24576-8192.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test_output_tiled/13_24576-8192.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/images/13_25600-116736.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/masks/13_25600-116736.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test_output_tiled/13_25600-116736.png" width="320" height="auto"></td>
</tr>

</table>

<hr>

We used the simple UNet Model <a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Follicular-Cell Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
Please cite these papers in your publications if it helps your research: 
<a href="https://www.sciencedirect.com/science/article/pii/S2667102621000036">
<b>pdf of Intelligent Medicine</b>
</a>
<pre>
@article{zhu2021hybrid,
  title={Hybrid model enabling highly efficient follicular segmentation in thyroid cytopathological whole slide image},
  author={Zhu, Chuang and Tao, Siyan and Chen, Huang and Li, Minzhen and Wang, Ying and Liu, Jun and Jin, Mulan},
  journal={Intelligent Medicine},
  year={2021},
  publisher={Elsevier}
}
</pre>
License: This data can be freely used for academic purposes. (non-commercial)
<br>

<h3>
<a id="2">
2 Follicular-Cell ImageMask Dataset
</a>
</h3>
 If you would like to train this Follicular-Cell Segmentation model by yourself,
 please download the Tiled dataset from the google drive 
<a href="https://drive.google.com/file/d/1ISpL9l-Dv8k93BrqGb0V0vaab9TrQP5U/view?usp=sharing">
Tiled-Follicular-Cell-ImageMask-Dataset-V2.zip</a>, 
 expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be

<pre>
./dataset
└─Tiled-Follicular-Cell-V2
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
 
 
<b>Tiled-Follicular-Cell Dataset V2 Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/Tiled-Follicular-Cell-ImageMask-Dataset-V2_Statistics.png" width="512" height="auto"><br>

As shown above, the number of images of train and valid dataset is not so large, but enough to use for our segmentation model.

<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/train_masks_sample.png" width="1024" height="auto">
<br>


<h3>
3. Train Tensorflow UNet Model
</h3>
 We trained Follicular-Cell TensorflowUNet Model by using the configuration file
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2 and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b>, a large <b>base_kernels</b> and a large <b>dilation</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
model          = "TensorflowUNet"
generator      = True
image_width    = 512
image_height   = 512
image_channels = 3
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Enabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = True
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Image color space conversion</b><br>
Enabled color space conversion.
<pre>
[image]
color_converter = "cv2.COLOR_BGR2HSV"
</pre>

<b>Mask blurring</b><br>
Enabled mask blurring.
<pre>
[mask]
blur      = True
blur_size = (5,5)
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer and epoch_change_tiledinfer callbacks.<br>
<pre>
[train]
epoch_change_infer      = True
epoch_change_infer_dir  = "./epoch_change_infer"
epoch_change_tiledinfer = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images        = 6
</pre>

By using these callbacks, on every epoch_change, the inference procedures can be called
 for 6 images in <b>mini_test</b> folder. These will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_tiled_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/epoch_change_tiledinfer.png" width="1024" height="auto"><br>
<br>
<br>
In this experiment, the training process was terminated at epoch 20.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/train_console_output_at_epoch_20.png" width="720" height="auto"><br>
<br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/eval/train_losses.png" width="520" height="auto"><br>
<br>


<h3>
4.Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Follicular-Cell.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/evaluate_console_output_at_epoch_20.png" width="720" height="auto">
<br>
The loss (bce_dice_loss) to this Follicular-Cell test dataset was not low, and dice_coef not high as shown below.
<br>
<pre>
loss,0.2749
dice_coef,0.6472
</pre>


<h2>
5. Inference
</h2>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-Follicular-Cell</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Follicular-Cell.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
Sample test images<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/mini_test_images.png" width="1024" height="auto"><br>
Sample test mask (ground_truth)<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/mini_test_masks.png" width="1024" height="auto"><br>

<br>
Inferred test masks<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>

<h2>
6. Tiled Inference
</h2>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Augmented-Follicular-Cell</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Follicular-Cell.<br>
<pre>
./4.tiled_infer.bat
</pre>


<br>
Tiled inferred test masks for images of 1024x1024 pixels<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>

<hr>
<b>Enlarged images and masks of 1024x1024 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/images/13_23552-121856.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/masks/13_23552-121856.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test_output_tiled/13_23552-121856.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/images/13_23552-198656.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/masks/13_23552-198656.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test_output_tiled/13_23552-198656.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/images/13_24576-7168.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/masks/13_24576-7168.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test_output_tiled/13_24576-7168.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/images/13_23552-87040.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/masks/13_23552-87040.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test_output_tiled/13_23552-87040.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/images/13_26624-6144.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test/masks/13_26624-6144.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Follicular-Cell-V2/mini_test_output_tiled/13_26624-6144.png" width="320" height="auto"></td>
</tr>3
</table>
<hr>
<br>

<h3>Reference</h3>
<b>1. Hybrid model enabling highly efficient follicular segmentation <br>
in thyroid cytopathological whole slide image </b><br>
Chuang Zhu, Siyan Tao, Huang Chen, Minzhen Li, Ying Wang, Jun Liu, Mulan Jin<br>

https://doi.org/10.1016/j.imed.2021.04.002<br>
<a href="https://www.sciencedirect.com/science/article/pii/S2667102621000036">https://www.sciencedirect.com/science/article/pii/S2667102621000036</a>
<br>
<br>
<b>2. Tiled-ImageMask-Dataset-Follicular-Cell</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Follicular-Cell">
https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Follicular-Cell
</a>

