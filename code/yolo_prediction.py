import cv2

def detect_products(frame, keyword,model):
    results = model.predict(frame)
    class_counts = {}
    keyword_found = False
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for result in results:
        boxes = result.boxes.xyxy  
        confidences = result.boxes.conf  
        class_ids = result.boxes.cls

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if confidence > 0.5:  
                class_name = model.names[int(class_id)].lower()

                if class_name == keyword:
                    keyword_found = True
                    x1, y1, x2, y2 = map(int, box) 
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  
                    label = f"{class_name} {confidence:.2f}"
                    font_scale = 1.0
                    thickness = 2
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    label_y1 = max(y1 - 10, 0)
                    label_y2 = label_y1 + label_size[1]
                    cv2.rectangle(frame_rgb, (x1, label_y1), (x1 + label_size[0], label_y2), (0, 255, 0), -1)
                    cv2.putText(frame_rgb, label, (x1, label_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
    return frame_rgb, keyword_found, class_counts
