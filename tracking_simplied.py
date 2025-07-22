# -*- coding: utf-8 -*-
import csv
import os
import sys
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import (DEVICE, INFERENCE_SIZE, IOU_THRESHOLD, KPTS_CONF,
                    MAX_OBJECT_CNT, PERSON_CONF, XMEM_CONFIG, YOLO_EVERY)
from inference.inference_utils import (add_new_classes_to_dict,
                                       generate_colors_dict,
                                       get_iou_filtered_yolo_mask_bboxes,
                                       merge_masks, overlay_mask_on_image)
from inference.interact.interactive_utils import torch_prob_to_numpy_mask
from tracker import Tracker
from pose_estimation import Yolov8PoseModel

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--camera_id', type=int, default=0,
                        required=False, help='Camera device ID (default: 0)')
    parser.add_argument(
        '--width', type=int, default=INFERENCE_SIZE[0], required=False, help='Inference width')
    parser.add_argument(
        '--height', type=int, default=INFERENCE_SIZE[1], required=False, help='Inference height')
    parser.add_argument('--frames_to_propagate', type=int,
                        default=None, required=False, help='Frames to propagate')
    parser.add_argument('--output_video_path', type=str, default=None,
                        required=False, help='Output video path to save')
    parser.add_argument('--device', type=str, default=DEVICE,
                        required=False, help='GPU id')
    parser.add_argument('--person_conf', type=float, default=PERSON_CONF,
                        required=False, help='YOLO person confidence')
    parser.add_argument('--kpts_conf', type=float, default=KPTS_CONF,
                        required=False, help='YOLO keypoints confidence')
    parser.add_argument('--iou_thresh', type=float, default=IOU_THRESHOLD,
                        required=False, help='IOU threshold to find new persons bboxes')
    parser.add_argument('--yolo_every', type=int, default=YOLO_EVERY,
                        required=False, help='Find new persons with YOLO every N frames')
    parser.add_argument('--output_path', type=str,
                        default='tracking_results.csv', required=False, help='Output filepath')

    args = parser.parse_args()

    if torch.cuda.device_count() > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

    # Use live camera feed instead of video file
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    df = pd.DataFrame(columns=['frame_id', 'person_id', 'x1', 'y1', 'x2', 'y2'])

    if args.output_video_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # Default to 30 FPS if camera doesn't provide FPS info
            fps = 30.0
        result = cv2.VideoWriter(args.output_video_path, cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v'), fps, (args.width, args.height))

    yolov8pose_model = Yolov8PoseModel(DEVICE, PERSON_CONF, KPTS_CONF)
    tracker = Tracker(XMEM_CONFIG, MAX_OBJECT_CNT, DEVICE)
    persons_in_video = False

    class_color_mapping = generate_colors_dict(MAX_OBJECT_CNT+1)

    current_frame_index = 0
    class_label_mapping = {}
    filtered_bboxes = []
    masks = None
    mask_bboxes_with_idx = []

    print(f"Starting live tracking from camera {args.camera_id}...")
    print("Press 'q' to quit")

    with torch.cuda.amp.autocast(enabled=True):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            if args.frames_to_propagate is not None and current_frame_index >= args.frames_to_propagate:
                break

            frame = cv2.resize(frame, (args.width, args.height),
                               interpolation=cv2.INTER_AREA)
            
            # Always run YOLO detection at specified interval
            if current_frame_index % args.yolo_every == 0:
                yolo_filtered_bboxes = yolov8pose_model.get_filtered_bboxes_by_confidence(frame)
                
                if len(yolo_filtered_bboxes) == 0:
                    # Fallback: Try to get raw detections with more reasonable confidence
                    try:
                        if hasattr(yolov8pose_model, 'model'):
                            raw_results = yolov8pose_model.model(frame)
                            if raw_results and len(raw_results) > 0 and len(raw_results[0].boxes) > 0:
                                boxes = raw_results[0].boxes
                                # Filter for person class (usually class 0) with reasonable confidence
                                person_mask = boxes.cls == 0  # Person class
                                conf_mask = boxes.conf > max(0.3, args.person_conf * 0.5)  # Use lower threshold
                                
                                if torch.any(person_mask & conf_mask):
                                    filtered_boxes = boxes.xyxy[person_mask & conf_mask]
                                    
                                    # Convert to the expected format
                                    yolo_filtered_bboxes = []
                                    for box in filtered_boxes:
                                        x1, y1, x2, y2 = box.cpu().numpy()
                                        yolo_filtered_bboxes.append([x1, y1, x2, y2])
                    except Exception as e:
                        pass
            else:
                # Reuse previous detections
                yolo_filtered_bboxes = []

            # Check if we have any persons in the frame
            if len(yolo_filtered_bboxes) > 0:
                persons_in_video = True

            # Process tracking if persons are detected or if we're already tracking
            if persons_in_video:
                # CASE 1: First detection in video
                if len(class_label_mapping) == 0 and len(yolo_filtered_bboxes) > 0:
                    mask = tracker.create_mask_from_img(
                        frame, yolo_filtered_bboxes, device='0')
                    unique_labels = np.unique(mask)
                    class_label_mapping = {
                        label: idx for idx, label in enumerate(unique_labels) if label != 0}
                    mask = np.array([class_label_mapping.get(label, 0)
                                    for label in mask.flat]).reshape(mask.shape)
                    prediction = tracker.add_mask(frame, mask)
                    masks = torch.tensor(torch_prob_to_numpy_mask(prediction)).unsqueeze(0)
                    mask_bboxes_with_idx = tracker.masks_to_boxes_with_ids(masks)
                
                # CASE 2: Additional/new persons detected
                elif len(filtered_bboxes) > 0:
                    mask = tracker.create_mask_from_img(
                        frame, filtered_bboxes, device='0')
                    unique_labels = np.unique(mask)
                    class_label_mapping = add_new_classes_to_dict(
                        unique_labels, class_label_mapping)
                    mask = np.array([class_label_mapping.get(label, 0)
                                    for label in mask.flat]).reshape(mask.shape)
                    
                    # Convert current masks to numpy for merging
                    current_masks_np = masks.squeeze(0).numpy() if masks is not None else None
                    
                    # Merge masks - ensure both are tensors
                    if current_masks_np is not None:
                        merged_mask = merge_masks(torch.tensor(current_masks_np), torch.tensor(mask))
                    else:
                        merged_mask = torch.tensor(mask)
                    
                    prediction = tracker.add_mask(frame, merged_mask.numpy())
                    masks = torch.tensor(torch_prob_to_numpy_mask(prediction)).unsqueeze(0)
                    mask_bboxes_with_idx = tracker.masks_to_boxes_with_ids(masks)
                    filtered_bboxes = []  # Reset after processing
                
                # CASE 3: Only predict (no new detections)
                else:
                    if masks is not None:
                        prediction = tracker.predict(frame)
                        masks = torch.tensor(torch_prob_to_numpy_mask(prediction)).unsqueeze(0)
                        mask_bboxes_with_idx = tracker.masks_to_boxes_with_ids(masks)
                    else:
                        mask_bboxes_with_idx = []

                # Update filtered bboxes if we have new YOLO detections
                if current_frame_index % args.yolo_every == 0 and masks is not None:
                    filtered_bboxes = get_iou_filtered_yolo_mask_bboxes(
                        yolo_filtered_bboxes, mask_bboxes_with_idx, iou_threshold=args.iou_thresh)
            else:
                # Reset tracking state if no persons detected
                mask_bboxes_with_idx = []
                masks = None

            # Prepare visualization - start with clean frame
            display_frame = frame.copy()
            
            # Draw only person ID bounding boxes from tracking
            if len(mask_bboxes_with_idx) > 0:
                for bbox in mask_bboxes_with_idx:
                    try:
                        # Handle different possible bbox formats
                        if isinstance(bbox, (list, tuple, np.ndarray)):
                            if len(bbox) >= 5:
                                person_id = int(bbox[0])
                                x1, y1, x2, y2 = int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])
                            elif len(bbox) >= 4:
                                person_id = 1  # Default ID
                                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                            else:
                                continue
                        else:
                            continue
                        
                        # Ensure coordinates are within frame bounds
                        x1 = max(0, min(x1, args.width - 1))
                        y1 = max(0, min(y1, args.height - 1))
                        x2 = max(0, min(x2, args.width - 1))
                        y2 = max(0, min(y2, args.height - 1))
                        
                        # Only draw if bbox is valid (has area)
                        if x2 > x1 and y2 > y1:
                            # Draw bounding box with green line
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw person ID with background for better visibility
                            label = f'{person_id}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            
                            # Draw background rectangle for text
                            cv2.rectangle(display_frame, 
                                        (x1 - 2, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0] + 4, y1 - 2), 
                                        (0, 255, 0), -1)
                            
                            # Draw text
                            cv2.putText(display_frame, label, (x1, y1 - 8), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                            
                    except Exception as e:
                        continue

            # Write to output video if enabled
            if args.output_video_path is not None:
                result.write(display_frame)

            # Show live results
            cv2.imshow('Person Tracking', display_frame)
            
            # Save tracking data to DataFrame
            if len(mask_bboxes_with_idx) > 0:
                for bbox in mask_bboxes_with_idx:
                    person_id = bbox[0]
                    x1 = bbox[1]
                    y1 = bbox[2]
                    x2 = bbox[3]
                    y2 = bbox[4]
                    df.loc[len(df.index)] = [
                        int(current_frame_index), int(person_id), x1, y1, x2, y2]
            else:
                df.loc[len(df.index)] = [int(current_frame_index),
                                         None, None, None, None, None]
            
            current_frame_index += 1
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    if args.output_video_path is not None:
        result.release()
    cv2.destroyAllWindows()
    df.to_csv(args.output_path, index=False)
    print(f"\nTracking complete. Results saved to {args.output_path}")