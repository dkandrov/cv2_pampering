import cv2
import mediapipe as mp
import math
import random
import datetime

max_num_hands = 6

cap = cv2.VideoCapture(0) # camera
hands = mp.solutions.hands.Hands(max_num_hands=max_num_hands) #AI object for hands detection
random.seed(datetime.datetime.now().timestamp())
colors = []
for _ in range(max_num_hands):
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)) # initial gloves color
    colors.append(color)

while True:
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'): # Close window
        break
    if key & 0xFF == ord('c'): # Change gloves color
        for i in range(max_num_hands):
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)) # initial gloves color
            colors[i] = color

    success, image = cap.read() # Reading the image from the camera
    image = cv2.flip(image, 1) # Reflect the image by x for a correct image
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to rgb
    results = hands.process(imageRGB) # Catch hands with mediapipe

    if results.multi_hand_landmarks:
        hands_counter = 0
        for handLms in results.multi_hand_landmarks:
            label = {
                'x': 0, 'y': 0, 'num_fingers': 0, 'scale': 1
            }
            all_points = [-1] * 21
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                all_points[id] = (cx, cy)
            thickness1 = math.sqrt(math.pow(all_points[5][0] - all_points[9][0], 2) + math.pow(all_points[5][1] - all_points[9][1], 2))
            thickness2 = math.sqrt(math.pow(all_points[7][0] - all_points[8][0], 2) + math.pow(all_points[7][1] - all_points[8][1], 2))
            thickness = int(1.1 * max(thickness1, thickness2))

            for i in range(2,5):
                cv2.line(image, all_points[i-1], all_points[i], colors[hands_counter],thickness)
            for i in range(6,9):
                cv2.line(image, all_points[i-1], all_points[i], colors[hands_counter],thickness)
            for i in range(10,13):
                cv2.line(image, all_points[i-1], all_points[i], colors[hands_counter],thickness)
            for i in range(14,17):
                cv2.line(image, all_points[i-1], all_points[i], colors[hands_counter],thickness)
            for i in range(18,21):
                cv2.line(image, all_points[i-1], all_points[i], colors[hands_counter],thickness)

            cv2.line(image, all_points[5], all_points[9], colors[hands_counter],thickness)
            cv2.line(image, all_points[9], all_points[13], colors[hands_counter],thickness)
            cv2.line(image, all_points[13], all_points[17], colors[hands_counter],thickness)
            cv2.line(image, all_points[1], all_points[5], colors[hands_counter],thickness)
            cv2.line(image, all_points[1], all_points[9], colors[hands_counter],thickness)
            cv2.line(image, all_points[1], all_points[13], colors[hands_counter],thickness)
            cv2.line(image, all_points[1], all_points[17], colors[hands_counter],thickness)
            cv2.line(image, all_points[0], all_points[5], colors[hands_counter],thickness)
            cv2.line(image, all_points[0], all_points[9], colors[hands_counter],thickness)
            cv2.line(image, all_points[0], all_points[13], colors[hands_counter],thickness)
            cv2.line(image, all_points[0], all_points[17], colors[hands_counter],thickness)
            cv2.line(image, all_points[0], all_points[1], colors[hands_counter],thickness)

            palm_width = math.sqrt(math.pow(all_points[5][0] - all_points[17][0], 2) + math.pow(all_points[5][1] - all_points[17][1], 2))
            dx = all_points[0][0] - all_points[5][0]
            dy = all_points[0][1] - all_points[5][1]
            additional_point = [all_points[0][0] - dx//2, all_points[0][1] - dy // 2]

            cv2.circle(image, additional_point, int(palm_width / 1.75), colors[hands_counter], cv2.FILLED)
            dx = all_points[0][0] - all_points[17][0]
            dy = all_points[0][1] - all_points[17][1]
            additional_point = [all_points[0][0] - dx//3, all_points[0][1] - dy // 3]
            cv2.circle(image, additional_point, int(palm_width / 1.75), colors[hands_counter], cv2.FILLED)
            additional_point = [all_points[0][0] - 3*dx//4, all_points[0][1] - 3*dy // 4]
            cv2.circle(image, additional_point, int(palm_width / 3), colors[hands_counter], cv2.FILLED)

            hands_counter += 1

    cv2.imshow("Hand", image) # Display resulting image
