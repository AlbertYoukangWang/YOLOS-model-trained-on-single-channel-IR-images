## Training and Validating YOLOS model on single-channel IR image dataset

**Programmer**: Albert Wang, Daudi Basuki

**TA Support by**: Jason Hughes *(Sincerely Appreciate his dedicated advice and support throughout every aspect of the project, and his answers to our questions that effectively gives us a clear direction for resolving our problems.*

**Last Update**: 21/12/2024

The main challenge of this project is that we are targetting `single-channel images`. To gain the maximum flexibility and contol over the training & validation process when handling single-channel images, we choose to use the `YolosForObjectDetection` class from hugging face transformer library, instead of usingh the less customizable interface of `ultralytics`.  
