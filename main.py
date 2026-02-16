import cv2
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

def detect_full_body_person(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Ошибка чтения изображения.")
        return []
    

    results = model.predict(img)
    
    detected_people = []
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        
        for box in boxes:
            cls_id = int(box.cls[0])
            
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты прямоугольника вокруг объекта
                
                height = y2 - y1
                width = x2 - x1
                
                aspect_ratio = height / width
                
        
                if aspect_ratio > 1.5:
                    detected_people.append((x1, y1, x2, y2))
                    
    return detected_people


if __name__ == "__main__":
    image_path = 'young-couple-walking-together-town_1303-11905.jpg'
    detections = detect_full_body_person(image_path)
    
    if len(detections) > 0:
        print(f"В изображении обнаружено {len(detections)} человек в полный рост.")
    else:
        print("Человек в полный рост не обнаружен.")
