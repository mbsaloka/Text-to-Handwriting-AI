import xml.etree.ElementTree as ET
import glob
import html
import numpy as np
from tqdm import tqdm
import pickle
import torch

# Get all XML files in current directory
dataset = '../data/labeled-lineStrokes'
xml_files = glob.glob(dataset + '/**/*.xml', recursive=True)
xml_files = sorted(xml_files)

selected_text = [
    'tip of hook. Catch long end of',
    'They say that our farmers',
    'Near the end of a long',
    'of the production of atomic',
    'say. Play an old pianna instead',
    'Guy Eden writes: Treasury',
    'structed in the precise functions of',
    'grim, sturdy house that was',
    'between 55 and 60 hours a',
    'The main provisions of a closure',
    'questions, such as birth control'
]

# Function to parse an individual XML file
def parse_xml_file(file_path):
    try:
        tree = ET.parse(file_path)
    except Exception as e:
        print('Error parsing', file_path, e)
        return None
    root = tree.getroot()

    # Extract text information
    text_element = root.find('.//Text')
    text_val = text_element.text if text_element is not None else ""

    # Check if text is in selected_text
    if text_val not in selected_text:
        return None

    # Handle HTML entities
    if text_val:
        text_val = html.unescape(text_val)

    # Extract Writer ID
    writer_element = root.find('.//Writer')
    writer_id = int(writer_element.attrib.get('writerID', 0)) if writer_element is not None else ""

    # Extract strokes
    strokes = []
    stroke_elements = root.findall('.//Stroke')
    for stroke_element in stroke_elements:
        points = []
        point_elements = stroke_element.findall('.//Point')
        for point_element in point_elements:
            try:
                x = float(point_element.attrib.get('x', 0))
                y = float(point_element.attrib.get('y', 0))
                time = float(point_element.attrib.get('time', 0))
                points.append((x, y, time))
            except Exception as e:
                print('Error reading point in', file_path, e)
                continue
        if points:
            strokes.append(points)

    return {
        'file': file_path,
        'writer_id': writer_id,
        'text': text_val,
        'strokes': strokes
    }

# Parse all XML files
all_data = []
for file in tqdm(xml_files, desc="Parsing XML files", unit="file"):
    parsed = parse_xml_file(file)
    if parsed is not None:
        all_data.append(parsed)


# Change stroke format from list(list(tuple)) to array(array) numpy
# Also remove time, and add one more dimension for boolean pen_status

converted_data = []
for data in tqdm(all_data, desc="Converting strokes"):
    stroke_points = []
    for stroke in data['strokes']:
        x_points, y_points, _ = zip(*stroke)
        pen_status = [0] * len(x_points)
        pen_status[-1] = 1  # Pen up after last point in stroke
        stroke_data = list(zip(pen_status, x_points, y_points))
        stroke_points.extend(stroke_data)

    stroke_points = np.array(stroke_points, dtype=np.float32)

    converted_data.append({
        'text': data['text'],
        'strokes': stroke_points
    })
all_data = converted_data


# Normalize stroke from absolute position to delta position
norm_factor = 20
all_normalized_data = []

for data in tqdm(all_data, desc="Processing Files", unit="file"):
    strokes = data['strokes']

    xy = strokes[:, 1:3]
    pen_status = strokes[1:, 0]

    delta_xy = xy[1:] - xy[:-1]
    delta_xy /= norm_factor  # Normalize

    normalized_strokes = np.hstack((pen_status.reshape(-1, 1), delta_xy))

    # Invert Y
    normalized_strokes[:, 2] = -normalized_strokes[:, 2]

    normalized_strokes = np.vstack(([0, 0, 0], normalized_strokes))

    all_normalized_data.append({
        'text': data['text'],
        'strokes': normalized_strokes
    })


def pad_strokes(strokes_list):
    # Find length and max length
    lengths = [s.shape[0] for s in strokes_list]
    max_length = np.max(lengths)
    num_strokes = len(strokes_list)

    # Create mask
    mask_shape = (num_strokes, max_length)
    mask = np.zeros(mask_shape, dtype=np.float32)

    # Pad strokes
    padded_shape = (num_strokes, max_length, 3)
    padded = np.zeros(padded_shape, dtype=np.float32)
    for i, length in enumerate(lengths):
        padded[i, :length] = strokes_list[i]
        mask[i, :length] = 1.

    return padded, mask

all_normalized_strokes = [data['strokes'] for data in all_normalized_data]
strokes_list = [torch.tensor(s, dtype=torch.float32) for s in all_normalized_strokes]
padded_strokes, stroke_masks = pad_strokes(strokes_list)


def valid_offset_normalization(mean, std, data):
    data[:, :, 1:] -= mean
    data[:, :, 1:] /= std
    return data

# Normalize the data
train_mean = [0.33291695, -0.00524508]
train_std = [1.9019104, 1.677278]

dataset = valid_offset_normalization(train_mean, train_std, padded_strokes)

def unpad_strokes(padded, mask):
    unpadded = []
    for i in range(padded.shape[0]):
        length = int(mask[i].sum())
        unpadded.append(padded[i, :length])
    return unpadded

upadded_strokes = unpad_strokes(dataset, stroke_masks)

all_text = [data['text'] for data in all_normalized_data]

# Save the converted data to a file
with open('../data/style_stroke.pkl', 'wb') as f:
    pickle.dump(upadded_strokes, f)
print('Saved converted data to style_stroke.pkl')

with open('../data/style_text.pkl', 'wb') as f:
    pickle.dump(all_text, f)
print('Saved converted text to style_text.pkl')
