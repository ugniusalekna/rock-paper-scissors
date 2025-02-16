import cv2 as cv
from contextlib import contextmanager


@contextmanager
def video_capture(device_id=0):
    cap = cv.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError("Can't access the webcam")
    try:
        yield cap
    finally:
        cap.release()
        cv.destroyAllWindows()


def crop_square(image):
    h, w, _ = image.shape
    min_dim = min(h, w)
    x, y = (w - min_dim) // 2, (h - min_dim) // 2
    return image[y:y + min_dim, x:x + min_dim]


def draw_text(frame, text='', pos=(10, 30), font=cv.FONT_HERSHEY_SIMPLEX, font_scale=0.5, 
              color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), alpha=0.5):
    if not text:
        return
    (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    
    overlay = frame.copy()
    cv.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline), bg_color, -1)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv.putText(frame, text, pos, font, font_scale, color, thickness, cv.LINE_AA)


def draw_hist(frame, probs, class_map, pos=(10, 120), height=150, text_h=30):
    x, y = pos
    max_prob = max(probs)
    bar_width = max([cv.getTextSize(f"{class_name}: {prob:.2f}", cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for class_name, prob in zip(class_map.keys(), probs)]) + 10
    width = bar_width * len(probs)
    
    for class_name, idx in class_map.items():
        prob = probs[idx]
        bar_height = int((prob / max_prob) * height)
        color = (43, 142, 242)
        cv.rectangle(frame, (x + idx * bar_width, y + height - bar_height), 
                     (x + (idx + 1) * bar_width, y + height), color, -1)
        cv.rectangle(frame, (x + idx * bar_width, y + height - bar_height), 
                     (x + (idx + 1) * bar_width, y + height), (0, 0, 0), 2)

    cv.rectangle(frame, (x - 5, y + height + 5), (x + width + 5, y + height + text_h + 5), (0, 0, 0), -1)
    
    for class_name, idx in class_map.items():
        prob = probs[idx]
        text = f"{class_name}: {prob:.2f}"
        text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = x + idx * bar_width + (bar_width - text_size[0]) // 2
        draw_text(frame, text=text, font_scale=0.6, pos=(text_x, y + height + text_h), bg_color=(0, 0, 0, 0))