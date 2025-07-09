import cv2, numpy as np, pandas as pd
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

data, labels = [], []
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret: break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            lm = np.array([(pt.x, pt.y, pt.z) for pt in hand.landmark]).flatten()
            data.append(lm)
            lbl = int(input("Enter gesture ID: "))  # e.g., 0,1,...
            labels.append(lbl)
    cv2.imshow("Data Collection", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release(); cv2.destroyAllWindows()
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('gesture_data.csv', index=False)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

df = pd.read_csv('gesture_data.csv')
X = df.drop('label', axis=1).values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

model.save('gesture_model.h5')
import cv2, numpy as np
import mediapipe as mp
import load_model

model = load_model('gesture_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def recognize(landmarks):
    lm = np.array([(pt.x, pt.y, pt.z) for pt in landmarks.landmark]).flatten()
    pred = model.predict(lm.reshape(1, -1))
    return np.argmax(pred), np.max(pred)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if not ret: break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            g, conf = recognize(hand)
            cv2.putText(img, f"Gesture {g} {conf:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Recognition", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release(); cv2.destroyAllWindows()
base = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3), include_top=False, weights='imagenet'
)
model = Sequential([base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
