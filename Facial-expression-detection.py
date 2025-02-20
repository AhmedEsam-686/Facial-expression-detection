import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        
        if isinstance(result, list):
            result = result[0]
            
        emotion = result.get('dominant_emotion', 'No Face')
    except Exception as e:
        print("Error:", e)
        emotion = "No Face"
    
    cv2.putText(frame, f"Emotion: {emotion}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Video Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
