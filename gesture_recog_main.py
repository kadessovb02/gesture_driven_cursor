import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

clicking = False
double_clicking = False
prev_x, prev_y = 0, 0

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            x = prev_x * 0.9 + x * 0.1
            y = prev_y * 0.9 + y * 0.1
            prev_x, prev_y = x, y

            screen_width, screen_height = pyautogui.size()
            pyautogui.moveTo(x * screen_width, y * screen_height)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5

            if distance < 0.1 and not clicking and not double_clicking:
                # Perform a left click
                pyautogui.mouseDown()
                clicking = True
            elif distance >= 0.1 and clicking:
                pyautogui.mouseUp()
                clicking = False

            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            distance_double = ((thumb_tip.x - middle_finger_tip.x) ** 2 + (thumb_tip.y - middle_finger_tip.y) ** 2) ** 0.5

            if distance_double < 0.1 and not double_clicking and not clicking:
                pyautogui.doubleClick()
                double_clicking = True
            elif distance_double >= 0.1 :
                double_clicking = False

    cv2.imshow('MediaPipe Hands', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
