import cv2

video = cv2.VideoCapture("mixkit-highway-with-cars-in-a-zenithal-aerial-shot-49820-hd-ready.mp4", )
haarcascade = cv2.CascadeClassifier("haarcascade_car.xml")

while video.isOpened():
    return_value, image = video.read()
    final_output = image
    if not return_value:
        break
    face_rect = haarcascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=3)
    for x, y, width, height in face_rect:
        cv2.rectangle(final_output, (x, y), (x+width, y+height), (0, 0, 0), 5)
    cv2.imshow("magic", final_output)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()