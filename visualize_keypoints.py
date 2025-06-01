# visualize_keypoints.py
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.animation import FuncAnimation

# MediaPipe Hands uses 21 keypoints
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),      # Thumb
    (0,5), (5,6), (6,7), (7,8),      # Index
    (5,9), (9,10), (10,11), (11,12), # Middle
    (9,13), (13,14), (14,15), (15,16), # Ring
    (13,17), (17,18), (18,19), (19,20), # Pinky
    (0,17) # Palm base
]

def plot_hand(ax, keypoints):
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_title("Hand Keypoints")
    ax.scatter(xs, ys, color='blue')
    for i, j in HAND_CONNECTIONS:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], 'black')

def visualize_keypoints_sequence(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    print("Keys:", data.files)

    for k in data.files:
        print(k, type(data[k]), getattr(data[k], 'shape', None))

    if 'keypoints' not in data:
        print(f"[Error] No 'keypoints' array in: {npz_path}")
        return

    sequence = data['keypoints']
    if sequence.shape == ():
        print(f"[Error] Empty keypoints array in: {npz_path}")
        return

    T = sequence.shape[0]
    sequence = sequence.reshape((T, 2, 21, 3))  # [T, hands, points, xyz]

    fig, ax = plt.subplots(figsize=(4, 4))

    def update(frame):
        ax.clear()
        for hand_id in range(2):
            keypoints = sequence[frame][hand_id]
            if not np.allclose(keypoints, 0):  # skip empty hand
                plot_hand(ax, keypoints[:, :2])

    ani = FuncAnimation(fig, update, frames=T, interval=100)
    plt.show()

def find_valid_npz(directory):
    for fname in os.listdir(directory):
        path = os.path.join(directory, fname)
        try:
            data = np.load(path, allow_pickle=True)
            if 'keypoints' in data and data['keypoints'].shape != ():
                print("✅ Found valid file:", fname, "| Shape:", data['keypoints'].shape)
                return path
        except Exception as e:
            print("❌", fname, "->", e)
    return None

if __name__ == "__main__":
    keypoint_dir = "data/keypoints"
    test_file = find_valid_npz(keypoint_dir)
    if test_file:
        visualize_keypoints_sequence(test_file)
    else:
        print("No valid .npz file found to visualize.")
