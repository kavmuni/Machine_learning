from ultralytics import YOLO

# Load a model
model = YOLO("C:/Users/muralidharan/PycharmProjects/VS_code_workspace/Machine_learning/Deep_learning/01032026_Yolo_imae_processing/Signature_detection/runs/detect/train2/weights/best.pt")  # load a signature-detection fine-tuned model

# Inference using the model
results = model.predict("https://ultralytics.com/assets/signature-s.mp4", conf=0.75)
print(results[0])
for result in results:
    if result.probs is not None:
        print("Classification probabilities:", result.probs)
        print(result)
# Process results list
"""
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
"""