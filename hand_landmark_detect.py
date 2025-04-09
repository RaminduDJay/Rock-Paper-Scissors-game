# import cv2
# import mediapipe as mp

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# # Open the webcam feed
# cap = cv2.VideoCapture(0)

# while True:
#     # Read frame from webcam
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally for a selfie-view display
#     frame = cv2.flip(frame, 1)
    
#     # Convert the BGR frame to RGB for MediaPipe processing
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame and get the hand landmarks
#     results = hands.process(rgb_frame)

#     # If hands are detected, draw the landmarks
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw landmarks on the frame
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Optionally, print the landmark positions
#             for id, lm in enumerate(hand_landmarks.landmark):
#                 h, w, c = frame.shape
#                 x, y = int(lm.x * w), int(lm.y * h)
#                 z = lm.z  # z is depth
#                 print(f"Landmark {id}: X: {x}, Y: {y}, Z: {z}")

#     # Display the frame with hand landmarks
#     cv2.imshow("Hand Landmark Detection", frame)

#     # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close the OpenCV window
# cap.release()
# cv2.destroyAllWindows()










import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define a function to recognize gestures
def recognize_gesture(hand_landmarks):
    # Based on landmarks, detect the gesture. We will use some basic checks for gesture recognition.
    # Note: You can improve this with a more advanced model or approach for real-time gesture recognition.

    # Check for 'Rock' (Closed fist)
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y:  # Thumb tip is lower than thumb base
        return "Rock"

    # Check for 'Paper' (Open hand)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[5].x and \
       hand_landmarks.landmark[8].x < hand_landmarks.landmark[9].x:  # Spread fingers
        return "Paper"

    # Check for 'Scissors' (Two fingers up)
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and \
       hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y:  # Two fingers up
        return "Scissors"

    return "Unknown"

# Function to determine the winner
def determine_winner(player1_gesture, player2_gesture):
    if player1_gesture == player2_gesture:
        return "It's a tie!"
    elif (player1_gesture == "Rock" and player2_gesture == "Scissors") or \
         (player1_gesture == "Scissors" and player2_gesture == "Paper") or \
         (player1_gesture == "Paper" and player2_gesture == "Rock"):
        return "Player 1 wins!"
    else:
        return "Player 2 wins!"

# Initialize score
player1_score = 0
player2_score = 0

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Track gestures of two players (if detected)
    player1_gesture = "Unknown"
    player2_gesture = "Unknown"

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks for each hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Recognize the gesture for each player
            if idx == 0:  # First hand (Player 1)
                player1_gesture = recognize_gesture(hand_landmarks)
            elif idx == 1:  # Second hand (Player 2)
                player2_gesture = recognize_gesture(hand_landmarks)

    # Determine winner based on gestures
    if player1_gesture != "Unknown" and player2_gesture != "Unknown":
        winner = determine_winner(player1_gesture, player2_gesture)
        print(f"Player 1: {player1_gesture} | Player 2: {player2_gesture} -> {winner}")

        # Update the score
        if winner == "Player 1 wins!":
            player1_score += 1
        elif winner == "Player 2 wins!":
            player2_score += 1

    # Display the score on the screen
    cv2.putText(frame, f"Player 1: {player1_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Player 2: {player2_score}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Rock Paper Scissors Game", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
