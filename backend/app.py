from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import cv2
import yaml

app = FastAPI()

# Load model and config
model_path = 'model_files/best.onnx'
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
data_yaml = 'model_files/data.yaml'
with open(data_yaml, 'r') as f:
    data_config = yaml.safe_load(f)
class_names = data_config['names']
input_size = (640, 640)

# Preprocess image
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize(input_size, Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Postprocess predictions
# def postprocess_predictions(outputs: np.ndarray, conf_thres=0.5, iou_thres=0.45):
#     boxes = outputs[0].transpose(1, 0)  # (8400, 7) -> (7, 8400)
#     scores = boxes[4:5].T  # Confidence scores
#     classes = boxes[5:].T  # Class probabilities
#     boxes = boxes[:4].T  # xywh boxes

#     print(f"Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Classes shape: {classes.shape}")
#     print(f"Boxes: {boxes[:5]}, Scores: {scores[:5]}, Classes: {classes[:5]}")

#     # Filter by confidence
#     mask = scores.max(axis=1) > conf_thres
#     boxes = boxes[mask]
#     scores = scores[mask].max(axis=1)
#     class_ids = classes[mask].argmax(axis=1)

#     print(f"Filtered Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Class IDs shape: {class_ids.shape}", mask)

#     # # NMS
#     # indices = cv2.dnn.NMSBoxes(
#     #     boxes.tolist(), scores.tolist(), conf_thres, iou_thres
#     # )
#     # print(f"NMS indices: {indices}")
#     # if indices is None or len(indices) == 0:
#     #     print("No indices returned from NMS.")
#     #     return []
#     # print(f"Indices shape: {indices.shape if isinstance(indices, np.ndarray) else 'N/A'}")

#     # if isinstance(indices, tuple):
#     #     indices = indices[0]
#     # indices = np.array(indices).flatten()

#     # results = []
#     # for idx in indices:
#     #     box = boxes[idx]
#     #     x, y, w, h = box
#     #     x1, y1 = x - w / 2, y - h / 2
#     #     x2, y2 = x + w / 2, y + h / 2
#     #     results.append({
#     #         'bbox': [float(x1), float(y1), float(x2), float(y2)],
#     #         'confidence': float(scores[idx]),
#     #         'class_id': int(class_ids[idx]),
#     #         'class_name': class_names[int(class_ids[idx])]
#     #     })
#     # print(f"Results: {results[:5]}")  # Print first 5 results for debugging
#     # if not results:
#     #     print("No valid predictions found.")
#     #     return []
#     # print(f"Number of predictions: {len(results)}")

#     # return results
#     # NMS
#     indices = cv2.dnn.NMSBoxes(
#         boxes.tolist(), scores.tolist(), conf_thres, iou_thres
#     )
    
#     # Debug print
#     print(f"Raw indices: {indices}, type: {type(indices)}")
    
#     # Handle empty results
#     if indices is None or len(indices) == 0:
#         print("No indices returned from NMS.")
#         return []

#     # Convert indices to the correct format
#     if isinstance(indices, tuple):
#         indices = indices[0]
#     if isinstance(indices, np.ndarray):
#         indices = indices.flatten()
    
#     # Ensure indices is a list and within bounds
#     indices = [i for i in indices if i < len(boxes)]
#     print(f"Filtered indices: {indices}")
    
#     if not indices:
#         print("No valid indices after bounds checking.")
#         return []

#     results = []
#     for idx in indices:
#         try:
#             box = boxes[idx]
#             x, y, w, h = box
#             x1, y1 = x - w / 2, y - h / 2
#             x2, y2 = x + w / 2, y + h / 2
#             results.append({
#                 'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                 'confidence': float(scores[idx]),
#                 'class_id': int(class_ids[idx]),
#                 'class_name': class_names[int(class_ids[idx])]
#             })
#         except IndexError as e:
#             print(f"Index error at idx {idx}: {str(e)}")
#             continue
#         except Exception as e:
#             print(f"Error processing box at idx {idx}: {str(e)}")
#             continue

#     return results

# def postprocess_predictions(outputs: np.ndarray, conf_thres=0.5, iou_thres=0.45):
#     boxes = outputs[0].transpose(1, 0)  # (8400, 7) -> (7, 8400)
#     scores = boxes[4:5].T  # Confidence scores
#     classes = boxes[5:].T  # Class probabilities
#     boxes = boxes[:4].T  # xywh boxes

#     print(f"Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Classes shape: {classes.shape}")
#     print(f"Boxes: {boxes[:5]}, Scores: {scores[:5]}, Classes: {classes[:5]}")

#     # Filter by confidence
#     mask = scores.max(axis=1) > conf_thres
#     boxes = boxes[mask]
#     scores = scores[mask].max(axis=1)
#     class_ids = classes[mask].argmax(axis=1)

#     print(f"Filtered Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Class IDs shape: {class_ids.shape}")

#     # Convert boxes to list and ensure they're regular Python lists
#     boxes_list = boxes.tolist()
#     scores_list = scores.tolist()

#     # NMS
#     indices = cv2.dnn.NMSBoxes(
#         boxes_list, scores_list, conf_thres, iou_thres
#     )
    
#     # Debug print
#     print(f"Raw indices: {indices}, type: {type(indices)}")
    
#     # Handle empty results
#     if indices is None or len(indices) == 0:
#         print("No indices returned from NMS.")
#         return []

#     # Convert indices to regular Python integers
#     if isinstance(indices, np.ndarray):
#         indices = [int(i) for i in indices.flatten()]
#     elif isinstance(indices, tuple):
#         indices = [int(i) for i in indices[0]]
    
#     # Ensure indices is a list and within bounds
#     indices = [i for i in indices if i < len(boxes_list)]
#     print(f"Filtered indices: {indices}")
    
#     if not indices:
#         print("No valid indices after bounds checking.")
#         return []

#     results = []
#     for idx in indices:
#         try:
#             box = boxes_list[idx]  # Use boxes_list instead of boxes
#             x, y, w, h = box
#             x1, y1 = x - w / 2, y - h / 2
#             x2, y2 = x + w / 2, y + h / 2
#             results.append({
#                 'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                 'confidence': float(scores_list[idx]),  # Use scores_list
#                 'class_id': int(class_ids[idx]),
#                 'class_name': class_names[int(class_ids[idx])]
#             })
#         except IndexError as e:
#             print(f"Index error at idx {idx}: {str(e)}")
#             continue
#         except Exception as e:
#             print(f"Error processing box at idx {idx}: {str(e)}")
#             continue

#     return results

def postprocess_predictions(outputs: np.ndarray, conf_thres=0.5, iou_thres=0.45):
    # Get output dimensions
    print(f"Raw output shape: {outputs.shape}")
    
    # Reshape output
    boxes = outputs[0].transpose(1, 0)  # (8400, 7) -> (7, 8400)
    print(f"Transposed shape: {boxes.shape}")
    
    # Split predictions
    scores = boxes[4:5].T  # Confidence scores
    classes = boxes[5:].T  # Class probabilities
    boxes = boxes[:4].T  # xywh boxes (first 4 values)
    
    print(f"After splitting - Boxes: {boxes.shape}, Scores: {scores.shape}, Classes: {classes.shape}")

    # Filter by confidence
    mask = scores.max(axis=1) > conf_thres
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask].max(axis=1)
    filtered_class_ids = classes[mask].argmax(axis=1)
    
    print(f"After filtering - Boxes: {filtered_boxes.shape}, Scores: {filtered_scores.shape}")
    print(f"Sample box: {filtered_boxes[0] if len(filtered_boxes) > 0 else 'No boxes'}")

    # Prepare for NMS
    nms_boxes = []
    for box in filtered_boxes:
        x, y, w, h = box
        x1 = float(x - w/2)
        y1 = float(y - h/2)
        x2 = float(x + w/2)
        y2 = float(y + h/2)
        nms_boxes.append([x1, y1, x2, y2])
    
    print(f"NMS boxes prepared: {len(nms_boxes)}")

    if not nms_boxes:
        return []

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        filtered_scores.tolist(),
        conf_thres,
        iou_thres
    )
    
    print(f"NMS returned indices: {indices}")

    # Process results
    results = []
    if len(indices) > 0:
        indices = indices.flatten()
        for idx in indices:
            try:
                box = filtered_boxes[idx]
                print(box, "iiiiiiiiiiibox")
                x, y, w, h = box
                print(x, y, w, h, "x, y, w, hsbox")
                results.append({
                    'bbox': [float(x-w/2), float(y-h/2), float(x+w/2), float(y+h/2)],
                    'confidence': float(filtered_scores[idx]),
                    'class_id': int(filtered_class_ids[idx]),
                    'class_name': class_names[int(filtered_class_ids[idx])]
                })
            except Exception as e:
                print(f"Error processing detection {idx}: {e}")
                continue

    print(f"Final results: {len(results)} detections")
    return results

# def postprocess_predictions(outputs: np.ndarray, conf_thres=0.5, iou_thres=0.45):
#     # First reshape output
#     # YOLOv8 outputs: [batch, num_boxes, num_classes + 5]
#     predictions = outputs[0].transpose(1, 0)  # [num_boxes, num_classes + 5]
    
#     # Separate boxes, scores, and classes
#     boxes = predictions[:, :4]  # x, y, w, h
#     scores = predictions[:, 4:5] * predictions[:, 5:]  # obj_conf * cls_conf
#     class_ids = np.argmax(predictions[:, 5:], axis=1)
#     confidences = np.max(scores, axis=1)

#     # Filter by confidence
#     mask = confidences > conf_thres
#     boxes = boxes[mask]
#     scores = confidences[mask]
#     class_ids = class_ids[mask]

#     # Convert boxes from xywh to xyxy format
#     boxes_xyxy = np.zeros_like(boxes)
#     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2  # x1
#     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2  # y1 
#     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2  # x2
#     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2  # y2

#     # Apply NMS
#     indices = cv2.dnn.NMSBoxes(
#         boxes_xyxy.tolist(),
#         scores.tolist(),
#         conf_thres,
#         iou_thres
#     )

#     results = []
#     if len(indices) > 0:
#         indices = indices.flatten()
#         for idx in indices:
#             box = boxes_xyxy[idx]
#             results.append({
#                 'bbox': [float(x) for x in box],
#                 'confidence': float(scores[idx]),
#                 'class_id': int(class_ids[idx]),
#                 'class_name': class_names[int(class_ids[idx])]
#             })

#     print(f"Raw output shape: {outputs.shape}")
#     print(f"Number of detections: {len(results)}")
#     print(f"Sample detection: {results[0] if results else 'No detections'}")

#     return results

# def postprocess_predictions(outputs: np.ndarray, conf_thres=0.25, iou_thres=0.45):  # Lowered confidence threshold
#     print(f"Processing output shape: {outputs.shape}")
    
#     # Reshape from (1, 7, 8400) to (8400, 7)
#     predictions = outputs[0].transpose(1, 0)
#     print(f"Reshaped predictions: {predictions.shape}")
    
#     # Extract coordinates and scores
#     boxes = predictions[:, :4]  # First 4 values are x, y, w, h
#     objectness = predictions[:, 4]  # 5th value is objectness score
#     class_scores = predictions[:, 5:]  # Remaining values are class scores
    
#     print(f"Found {len(boxes)} potential boxes")
#     print(f"Class scores shape: {class_scores.shape}")
    
#     # Calculate confidence scores
#     scores = objectness[:, np.newaxis] * class_scores
#     class_ids = np.argmax(class_scores, axis=1)
#     confidences = np.max(scores, axis=1)
    
#     # Filter by confidence
#     mask = confidences > conf_thres
#     boxes = boxes[mask]
#     scores = confidences[mask]
#     class_ids = class_ids[mask]

#     print("Available classes:", class_names)
    
#     print(f"After confidence filtering: {len(boxes)} boxes")
#     if len(boxes) > 0:
#         print(f"Sample scores: {scores[:5]}")
#         print(f"Sample class_ids: {class_ids[:5]}")
    
#     # Convert boxes to xyxy format for NMS
#     boxes_xyxy = np.zeros_like(boxes)
#     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2  # x1
#     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2  # y1 
#     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2  # x2
#     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2  # y2
    
#     # Apply NMS
#     if len(boxes_xyxy) > 0:
#         indices = cv2.dnn.NMSBoxes(
#             boxes_xyxy.tolist(),
#             scores.tolist(),
#             conf_thres,
#             iou_thres
#         )
        
#         print(f"After NMS: {len(indices) if indices is not None else 0} boxes")
        
#         results = []
#         if indices is not None and len(indices) > 0:
#             indices = indices.flatten()
#             for idx in indices:
#                 try:
#                     box = boxes_xyxy[idx]
#                     results.append({
#                         'bbox': [float(x) for x in box],
#                         'confidence': float(scores[idx]),
#                         'class_id': int(class_ids[idx]),
#                         'class_name': class_names[int(class_ids[idx])]
#                     })
#                 except Exception as e:
#                     print(f"Error processing box {idx}: {str(e)}")
#                     continue
                
#         return results
    
#     return []

# def postprocess_predictions(outputs: np.ndarray, conf_thres=0.1, iou_thres=0.45):  # Lowered confidence threshold more
#     print(f"Processing output shape: {outputs.shape}")
    
#     # Reshape predictions
#     predictions = outputs[0].transpose(2, 1, 0)  # reshape to (8400, 7)
    
#     # Extract coordinates, objectness and class scores
#     boxes = predictions[..., :4]  # x, y, w, h
#     objectness = predictions[..., 4]  # object confidence
#     class_scores = predictions[..., 5:]  # class scores
    
#     # Calculate total confidence
#     scores = objectness[..., None] * class_scores
    
#     # Get best class and confidence for each detection
#     class_ids = np.argmax(class_scores, axis=1)
#     confidences = np.max(scores, axis=1)
    
#     # Count all potential objects (before filtering)
#     total_potential_objects = len(boxes)
#     print(f"Total potential objects detected: {total_potential_objects}")
    
#     # Filter by confidence
#     mask = confidences > conf_thres
#     boxes = boxes[mask]
#     scores = confidences[mask]
#     class_ids = class_ids[mask]
    
#     # Count objects per class
#     class_counts = {}
#     for class_id in class_ids:
#         class_name = class_names[int(class_id)]
#         class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
#     print("Objects detected per class (before NMS):", class_counts)
    
#     if len(boxes) == 0:
#         return {
#             'detections': [],
#             'statistics': {
#                 'total_potential_objects': total_potential_objects,
#                 'objects_per_class': {},
#                 'total_filtered_objects': 0
#             }
#         }
    
#     # Convert boxes to xyxy format for NMS
#     boxes_xyxy = np.zeros_like(boxes)
#     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2  # x1
#     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2  # y1 
#     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2  # x2
#     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2  # y2
    
#     # Apply NMS
#     indices = cv2.dnn.NMSBoxes(
#         boxes_xyxy.tolist(),
#         scores.tolist(),
#         conf_thres,
#         iou_thres
#     )
    
#     # Process results
#     results = []
#     final_class_counts = {}
    
#     if indices is not None and len(indices) > 0:
#         indices = indices.flatten()
#         for idx in indices:
#             try:
#                 box = boxes_xyxy[idx]
#                 class_id = int(class_ids[idx])
#                 class_name = class_names[class_id]
                
#                 # Update final counts
#                 final_class_counts[class_name] = final_class_counts.get(class_name, 0) + 1
                
#                 results.append({
#                     'bbox': [float(x) for x in box],
#                     'confidence': float(scores[idx]),
#                     'class_id': class_id,
#                     'class_name': class_name
#                 })
#             except Exception as e:
#                 print(f"Error processing box {idx}: {str(e)}")
#                 continue
    
#     print("Final object counts after NMS:", final_class_counts)
    
#     return {
#         'detections': results,
#         'statistics': {
#             'total_potential_objects': total_potential_objects,
#             'objects_per_class': final_class_counts,
#             'total_filtered_objects': len(results)
#         }
#     }

# def postprocess_predictions(outputs: np.ndarray, conf_thres=0.1, iou_thres=0.45):
#     print(f"Processing output shape: {outputs.shape}")
    
#     # Reshape predictions - YOLOv8 output is (1, 7, 8400)
#     predictions = outputs[0]  # Remove batch dimension
#     predictions = predictions.T  # Transpose to (8400, 7)
    
#     # Extract coordinates, objectness and class scores
#     boxes = predictions[:, :4]  # x, y, w, h
#     objectness = predictions[:, 4]  # object confidence
#     class_scores = predictions[:, 5:]  # class scores
    
#     print(f"Shapes after split - Boxes: {boxes.shape}, Objectness: {objectness.shape}, Class scores: {class_scores.shape}")
    
#     # Calculate total confidence and get best class
#     confidences = objectness * np.max(class_scores, axis=1)
#     class_ids = np.argmax(class_scores, axis=1)
    
#     # Count all potential objects (before filtering)
#     total_potential_objects = len(boxes)
#     print(f"Total potential objects detected: {total_potential_objects}")
    
#     # Filter by confidence
#     mask = confidences > conf_thres
#     boxes = boxes[mask]
#     confidences = confidences[mask]
#     class_ids = class_ids[mask]
    
#     # Count objects per class
#     class_counts = {}
#     for class_id in class_ids:
#         class_name = class_names[int(class_id)]
#         class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
#     print("Objects detected per class (before NMS):", class_counts)
    
#     if len(boxes) == 0:
#         return {
#             'detections': [],
#             'statistics': {
#                 'total_potential_objects': total_potential_objects,
#                 'objects_per_class': {},
#                 'total_filtered_objects': 0
#             }
#         }
    
#     # Convert boxes to xyxy format for NMS
#     boxes_xyxy = np.zeros_like(boxes)
#     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2  # x1
#     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2  # y1 
#     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2  # x2
#     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2  # y2
    
#     # Apply NMS
#     indices = cv2.dnn.NMSBoxes(
#         boxes_xyxy.tolist(),
#         confidences.tolist(),
#         conf_thres,
#         iou_thres
#     )
    
#     # Process results
#     results = []
#     final_class_counts = {}
    
#     if indices is not None and len(indices) > 0:
#         indices = indices.flatten()
#         for idx in indices:
#             try:
#                 box = boxes_xyxy[idx]
#                 class_id = int(class_ids[idx])
#                 class_name = class_names[class_id]
                
#                 # Update final counts
#                 final_class_counts[class_name] = final_class_counts.get(class_name, 0) + 1
                
#                 results.append({
#                     'bbox': [float(x) for x in box],
#                     'confidence': float(confidences[idx]),
#                     'class_id': class_id,
#                     'class_name': class_name
#                 })
#             except Exception as e:
#                 print(f"Error processing box {idx}: {str(e)}")
#                 continue
    
#     print("Final object counts after NMS:", final_class_counts)
    
#     return {
#         'detections': results,
#         'statistics': {
#             'total_potential_objects': total_potential_objects,
#             'objects_per_class': final_class_counts,
#             'total_filtered_objects': len(results)
#         }
#     }



@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        print(f"Received file: {file.filename}, size: {len(contents)} bytes")
        if not contents:
            return JSONResponse(content={'error': 'Empty file'}, status_code=400)
        # Open image
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        print(f"Image size: {image.size}, mode: {image.mode}")

        # Preprocess
        input_data = preprocess_image(image)
        print(f"Input data shape: {input_data.shape}")
        if input_data.shape[1:] != (3, *input_size):
            return JSONResponse(content={'error': 'Invalid image dimensions'}, status_code=400)
        print(f"Input data type: {input_data.dtype}, range: {input_data.min()} - {input_data.max()}")

        # Inference
        input_name = session.get_inputs()[0].name
        print(f"Input name: {input_name}")
        if input_name is None:
            return JSONResponse(content={'error': 'Model input name not found'}, status_code=500)
        print("Running inference...")
        outputs = session.run(None, {input_name: input_data})[0]
        print(f"Outputs shape: {outputs.shape}")
        if outputs is None or len(outputs) == 0:
            return JSONResponse(content={'error': 'No predictions made'}, status_code=500)
        print("Inference completed successfully.")
        print(f"Outputs type: {type(outputs)}, shape: {outputs.shape}, dtype: {outputs.dtype}")

        # Postprocess
        predictions = postprocess_predictions(outputs)
        print(f"Predictions: {predictions}")
        if not predictions:
            return JSONResponse(content={'predictions': []}, status_code=200)
        # print(f"Number of predictions: {len(predictions)}")

        return JSONResponse(content={'predictions': predictions})
        # return JSONResponse(content={
        #     'predictions': predictions['detections'],
        #     'statistics': predictions['statistics']
        # })
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

@app.get('/health')
async def health():
    return {'status': 'healthy'}






















# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import onnxruntime as ort
# import numpy as np
# from PIL import Image
# import io
# import cv2
# import yaml

# app = FastAPI()

# # Load model and config
# model_path = '/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/shelfsense_yolov8n_final/weights/best.onnx'
# session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# data_yaml = '/content/drive/MyDrive/ShelfsenseAI.v1i.yolov11/data.yaml'
# with open(data_yaml, 'r') as f:
#     data_config = yaml.safe_load(f)
# class_names = data_config['names']
# input_size = (640, 640)

# # Preprocess image
# def preprocess_image(image: Image.Image) -> np.ndarray:
#     img = image.resize(input_size, Image.Resampling.LANCZOS)
#     img = np.array(img).astype(np.float32) / 255.0
#     img = img.transpose(2, 0, 1)  # HWC to CHW
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # Postprocess predictions
# def postprocess_predictions(outputs: np.ndarray, conf_thres=0.5, iou_thres=0.45):
#     boxes = outputs[0].transpose(1, 0)  # (8400, 7) -> (7, 8400)
#     scores = boxes[4:5].T  # Confidence scores
#     classes = boxes[5:].T  # Class probabilities
#     boxes = boxes[:4].T  # xywh boxes

#     # Filter by confidence
#     mask = scores.max(axis=1) > conf_thres
#     boxes = boxes[mask]
#     scores = scores[mask].max(axis=1)
#     class_ids = classes[mask].argmax(axis=1)

#     # NMS
#     indices = cv2.dnn.NMSBoxes(
#         boxes.tolist(), scores.tolist(), conf_thres, iou_thres
#     )
#     if isinstance(indices, tuple):
#         indices = indices[0]
#     indices = np.array(indices).flatten()

#     results = []
#     for idx in indices:
#         box = boxes[idx]
#         x, y, w, h = box
#         x1, y1 = x - w / 2, y - h / 2
#         x2, y2 = x + w / 2, y + h / 2
#         results.append({
#             'bbox': [float(x1), float(y1), float(x2), float(y2)],
#             'confidence': float(scores[idx]),
#             'class_id': int(class_ids[idx]),
#             'class_name': class_names[int(class_ids[idx])]
#         })
#     return results

# @app.post('/predict')
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read image
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert('RGB')

#         # Preprocess
#         input_data = preprocess_image(image)

#         # Inference
#         input_name = session.get_inputs()[0].name
#         outputs = session.run(None, {input_name: input_data})[0]

#         # Postprocess
#         predictions = postprocess_predictions(outputs)

#         return JSONResponse(content={'predictions': predictions})
#     except Exception as e:
#         return JSONResponse(content={'error': str(e)}, status_code=500)

# @app.get('/health')
# async def health():
#     return {'status': 'healthy'}