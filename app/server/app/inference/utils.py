import numpy as np
import matplotlib.pyplot as plt

class Global:
    train_mean = [0.33291695, -0.00524508]
    train_std = [1.9019104, 1.677278]
    char_to_index = {
        ' ': 0, '!': 1, '"': 2, '#': 3, '%': 4, '&': 5, "'": 6, '(': 7, ')': 8,
        '+': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '2': 16,
        '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24,
        ';': 25, '?': 26, 'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 'F': 32,
        'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 'N': 40,
        'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 'U': 47, 'V': 48,
        'W': 49, 'X': 50, 'Y': 51, 'Z': 52, 'a': 53, 'b': 54, 'c': 55, 'd': 56,
        'e': 57, 'f': 58, 'g': 59, 'h': 60, 'i': 61, 'j': 62, 'k': 63, 'l': 64,
        'm': 65, 'n': 66, 'o': 67, 'p': 68, 'q': 69, 'r': 70, 's': 71, 't': 72,
        'u': 73, 'v': 74, 'w': 75, 'x': 76, 'y': 77, 'z': 78
    }

def delta_to_absolute(strokes):
    abs_coords = []
    x, y = 0, 0
    for pen_status, dx, dy in strokes:
        x += dx
        y += dy
        abs_coords.append((pen_status, x, y))
    return np.array(abs_coords)

def data_denormalization(mean, std, data):
    data[:, :, 1:] *= std
    data[:, :, 1:] += mean
    return data

def visualize_handwriting(strokes, invert_y=True):
    plt.figure(figsize=(12, 6))
    plt.title('Handwriting Visualization', fontsize=20, pad=15)
    plt.xlabel('X Coordinate', fontsize=16, labelpad=10)
    plt.ylabel('Y Coordinate', fontsize=16, labelpad=10)

    colors = ['#766CDB', '#DA847C', '#D9CC8B', '#7CD9A5', '#877877', '#52515E']
    xs, ys = [], []

    for i, point in enumerate(strokes):
        pen_status, x, y = point
        xs.append(x)
        ys.append(y)
        if pen_status == 1:
            plt.plot(xs, ys, color=colors[i % len(colors)], linewidth=2)
            xs, ys = [], []

    if invert_y:
        plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    plt.tight_layout()
    plt.show()
