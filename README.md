## Training and Validating YOLOS model on single-channel IR image dataset

**Programmer**: Albert Wang, Daudi Basuki

**TA Support by**: Jason Hughes *(Sincerely Appreciate his dedicated advice and support throughout every aspect of the project, and his answers to our questions that effectively gives us a clear direction for resolving our problems.*

**Code Last Update**: 21/12/2024

**Readme Last Update**: 21/12/2024

<br>

> The main challenge of this project is that we are targeting `single-channel images`. To gain the maximum flexibility and contol over the training & validation process when handling single-channel images, we choose to use the `YolosForObjectDetection` class from hugging face transformer library, instead of using the less customizable interface of `ultralytics`.  

**Major Work Done so far/ Functionalities of our code:**

Code for Both Training and Validation has been fully fixed. I have conducted 2 training experiments: `runs/experiment_1` training on 100% data, `runs/experiment_2_successful_val` training on 90% data, while performing validation on the remaining 10% data. In order to see *tensorboard visualization* of results for both experiments, clone the directory, open `Tensorboard Visualization.ipynb` file and runs the corresponding cells.

<br>

**Documentation of major bugs/problems that have been fixed:**

1. Custom `collate_fn` function. Default one doesn't work because while batched `image` is a `tensor`, batched `target (label)` is a `list`. Default collate functions use `stack`, our custom `collate_fn` uses `append` to handle the lists. Especially notice that `target (label)` here is a `dictionary` with 2 keys `"class_labels"` & `"boxes"`

2. (**Important**) I changed bounding box specification from (xmax, ymax, xmin, ymin) -> (xmin, ymin, xmax, ymax) when loading our image dataset. This is to accomodate the standard specification of the function `post_process_object_detection`.

3. Handle the case when there's only 1 bounding box detected in ground truth / predicted images. Because usually the dimensionality will only be 1 instead of 2 in thus case.

4. Data Type of loaded label has to strictly fit to the description in the documentation. Especially `bounding box` has to be changed from `float64` to `float32`, and `label` needs to be of type `long` as well.

5. Since we are targeting `single-channel images`, `YolosImageProcessor` should be adjusted from its default 3-channel behaviour during instantiation. Specifically, I set `image_mean = [0.5]`, `image_std = [0.5]` (Value could be changed later). Otherwise, you'll get a value error: `ValueError: mean must have 1 elements if it is an iterable, got 3`

6. IoU Calculation (**Still need further update!**). To handle the case when the number of predicted boxes and ground truth are different, we need to match the different boxes and take the mean IoU across all matches. And because the predicted bounding boxes are often too many, we use `IoU_threshold` and `confidence_score` to filter out less important ones. However, this method still proves to be problematic because we observe that the IoU we obtain during this process is significantly smaller than normal. I think the cause of the problem is that some images have 0 detected boxes (due to the filtering) and their IoU is set to 0.0. As we take the mean across all IoU calculations, the IoU score is significantly pulled down. Further update of the IoU calculation method /algorithm shall be implemented in the following work.

7. Other bugs that were fixed by carefully checking the documentation of input types and shapes. For example, `pred_boxes` and `true_boxes` have to be `numpy arrays` instead of `pytorch tensors` when calculating `IoU`. 
