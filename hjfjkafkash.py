import cv2
import mediapipe as mp
mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.75, min_tracking_confidence=0.75, max_num_hands=2)
camera = cv2.VideoCapture(0)
from google.protobuf.json_format import MessageToDict

while True:
    return_value, image = camera.read()
    image = cv2.flip(image, 1)
    if return_value == False:
        break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        #check if both hands are inside the frame
        if len(results.multi_hand_landmarks) == 2:
            cv2.putText(image, "Both Hands Detected", (10, 10), cv2.FONT_HERSHEY_PLAIN, 5.5, (0, 0, 0), 5)
        else:
            for i in results.multi_handedness:
                text = MessageToDict(i)["classification"][0]["label"]
                if text == "Left":
                    cv2.putText(image, "Left Hand Detected", (10, 10), cv2.FONT_HERSHEY_PLAIN, 5.5, (0, 0, 0), 5)
                elif text == "Right":
                    cv2.putText(image, "Right Hand Detected", (10, 10), cv2.FONT_HERSHEY_PLAIN, 5.5, (0, 0, 0), 5)
    if cv2.waitKey(0) == 27:
        break
                    

    cv2.imshow("Image", image)
cv2.destroyAllWindows()