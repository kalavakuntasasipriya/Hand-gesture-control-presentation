import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import time
import numpy as np
from math import hypot

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Timestamp variables for delays
last_next_slide_time = 0
last_prev_slide_time = 0
last_click_time = 0
last_zoom_time = 0
last_scroll_time = 0

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Get the landmarks
        lmList = hand['lmList']
        if lmList:
            x1, y1 = lmList[4][:2]  # Thumb tip
            x2, y2 = lmList[8][:2]  # Index finger tip
            x3, y3 = lmList[12][:2]  # Middle finger tip

            current_time = time.time()

            # Move to next slide if only the thumb is raised, with a 1.5-second delay
            if fingers == [1, 0, 0, 0, 0] and current_time - last_next_slide_time > 1.5:
                pyautogui.hotkey('right')
                last_next_slide_time = current_time

            # Move to previous slide if only the pinky is raised, with a 1.5-second delay
            elif fingers == [0, 0, 0, 0, 1] and current_time - last_prev_slide_time > 1.5:
                pyautogui.hotkey('left')
                last_prev_slide_time = current_time

            # Cursor control based on the index finger’s position
            elif fingers == [0, 1, 0, 0, 0]:  # Only index finger is raised
                screen_w, screen_h = pyautogui.size()
                cursor_x = np.interp(x2, [0, img.shape[1]], [0, screen_w])
                cursor_y = np.interp(y2, [0, img.shape[0]], [0, screen_h])
                pyautogui.moveTo(cursor_x, cursor_y)

            # Click based on thumb-middle finger distance, with a 0.5-second delay
            thumb_middle_distance = hypot(x3 - x1, y3 - y1)
            if thumb_middle_distance < 20 and current_time - last_click_time > 0.5:
                pyautogui.click()
                last_click_time = current_time

            elif 20 <= thumb_middle_distance < 30 and current_time - last_click_time > 0.5:
                pyautogui.doubleClick()
                last_click_time = current_time

            # Zoom In/Out based on thumb-index finger distance, with a 0.3-second delay
            thumb_index_distance = hypot(x2 - x1, y2 - y1)
            if fingers == [1, 1, 0, 0, 0] and current_time - last_zoom_time > 0.3:
                if thumb_index_distance > 150:
                    pyautogui.hotkey('ctrl', '+')
                    last_zoom_time = current_time
                elif thumb_index_distance < 90:
                    pyautogui.hotkey('ctrl', '-')
                    last_zoom_time = current_time

            # Scroll if both index and middle fingers are raised, with a 0.3-second delay
            if fingers == [0, 1, 1, 0, 0] and current_time - last_scroll_time > 0.3:
                pyautogui.scroll(-30)  # Scroll down
                last_scroll_time = current_time
            elif fingers == [0, 1, 1, 0, 0] and current_time - last_scroll_time > 0.3:
                pyautogui.scroll(30)  # Scroll up
                last_scroll_time = current_time

    # Display the video feed
    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
