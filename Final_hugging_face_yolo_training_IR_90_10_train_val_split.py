# Last Update: 20/12/2024
# main program for training and validating our YOLOS model on single-channel IR image dataset
# Tensorboard is also implemented to keep track of Train loss, Validation loss, and Validation IoU.  

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam 
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from transformers import YolosConfig, YolosImageProcessor, YolosForObjectDetection
from Final_custom_dataset import FinalCustomDataset
from Final_IoU_util_functions import calculate_iou
import numpy as np

def custom_collate_fn(batch):
    # Separate images and targets
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Stack images into batches
    batch_images = torch.stack(images, dim=0)
    
    batch = [] # batch can only be a list instead of dictionaries, otherwise you get weird errors!
    batch.append(batch_images)
    batch.append(targets) # "targets" needs to be a list, thus default "stack" doesn't work. custom collate_fn uses "append" instead
    
    return batch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    IoU_threshold = 0.7
    batch_size = 8

    image_dir = "/mnt/extra-dtc/Infrared-Object-Detection/datasets/infrared/images"
    label_dir = "/mnt/extra-dtc/Infrared-Object-Detection/datasets/infrared/labels"
    dataset = FinalCustomDataset(image_dir, label_dir)
    
    # Split dataset into 90% train and 10% validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    config = YolosConfig()
    config.num_channels = 1

    # Targeting 1-dim image
    processor = YolosImageProcessor(do_rescale=False, image_mean = [0.5], image_std = [0.5])

    model = YolosForObjectDetection(config=config)
    optimizer = Adam(model.parameters(), lr=1e-4)

    model.to(device)

    writer = SummaryWriter(log_dir='./runs/experiment_2_successful_val')

    epochs = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for step, (image, label) in enumerate(train_dataloader):
            inputs = processor(image, return_tensors="pt").to(device)
            label_onGPU = []
            for batch_label in label:
                label_onGPU.append({
                    'class_labels': batch_label['class_labels'].to(device).long(),
                    'boxes': batch_label['boxes'].to(device).float()
                })

            # print(label_onGPU)
            outputs = model(**inputs, labels=label_onGPU)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + step)

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss:.4f}")

        # Perform Validation after each epoch
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            # Take a batch of validation data at a time
            for image, label in val_dataloader:
                print("\nWork on Another Validation Batch!")
                inputs = processor(image, return_tensors="pt").to(device)
                label_onGPU = []
                for batch_label in label:
                    label_onGPU.append({
                        'class_labels': batch_label['class_labels'].to(device).long(),
                        'boxes': batch_label['boxes'].to(device).float()
                    })

                outputs = model(**inputs, labels=label_onGPU)
                loss = outputs.loss
                val_loss += loss.item() # Obtain validation loss

                # Post-processing - convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
                # print(f"**********************{image.size()}")
                # target_sizes = torch.tensor([image.size()[::-1]])
                results = processor.post_process_object_detection(outputs, threshold=IoU_threshold) # Need to get rid of "[0]" in sample code

                # Obtain confidence score, label & box from the predictions
                for i, result in enumerate(results):
                    print(f"Validating Image {i}")
                    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                        box = [round(j, 2) for j in box.tolist()]
                        print(f"For Image {i}, Detected Class {label.item()} with confidence {round(score.item(), 3)} at location {box}")

                # Prepare pred_boxes & true_boxes for IoU calculation
                for result, sample_n_in_batch_label in zip(results, label_onGPU):
                    true_boxes = sample_n_in_batch_label["boxes"].cpu().numpy()
                    true_boxes = np.expand_dims(true_boxes, axis=0) if true_boxes.ndim == 1 else true_boxes # Handle the case: "boxes" only has 1 object

                    pred_boxes = result["boxes"].cpu().numpy()
                    pred_boxes = np.expand_dims(pred_boxes, axis=0) if pred_boxes.ndim == 1 else pred_boxes # Handle the case: "boxes" only has 1 object
                    
                    # print(pred_boxes)
                    # print(true_boxes)

                    # Calculate IoU for validation
                    iou = calculate_iou(pred_boxes, true_boxes, IoU_threshold)
                    val_iou += iou

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_iou = val_iou / (len(val_dataloader) * batch_size)
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f}, Validation IoU: {avg_val_iou:.4f}")

        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('IoU/val', avg_val_iou, epoch)

    print("Training Complete")

    # Save the trained model's state_dict
    torch.save(model.state_dict(), "yolos_model_successful_val.pth")