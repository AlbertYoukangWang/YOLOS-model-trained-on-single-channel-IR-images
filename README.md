## Training and Validating YOLOS model on single-channel IR image dataset

**Programmer**: Albert Wang, Daudi Basuki

**TA Support by**: Jason Hughes *(Sincerely Appreciate his dedicated advice and support throughout every aspect of the project, and his answers to our questions that effectively gives us a clear direction for resolving our problems.*

**Last Update**: 21/12/2024

<br>

> The main challenge of this project is that we are targeting `single-channel images`. To gain the maximum flexibility and contol over the training & validation process when handling single-channel images, we choose to use the `YolosForObjectDetection` class from hugging face transformer library, instead of using the less customizable interface of `ultralytics`.  

**Major Work Done so far/ Functionalities of our code:**

Code for Both Training and Validation has been fully fixed. We have conducted 2 training experiments: `runs/experiment_1` training on 100% data, `runs/experiment_2_successful_val` training on 90% data, while performing validation on the remaining 10% data. In order to see *tensorboard visualization* of results for both experiments, clone the directory, open `Tensorboard Visualization.ipynb` file and runs the corresponding cells.

**Documentation of major bugs/problems fixed:**

1. I wrote a custom `collate_fn` function. Default one doesn't work because 
