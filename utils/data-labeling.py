import os
from datetime import datetime
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Root paths
original_root = "../data/original/"
line_root = "../data/lineStrokes/"
output_root = "../data/labeled-lineStrokes/"
log_folder = "log/"
log_file_name = f"labeling_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = os.path.join(log_folder, log_file_name)

# Create output and log folders if they do not exist
os.makedirs(output_root, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

# Initialize log file
with open(log_file, "w") as log:
    log.write("Labeling Dataset Log\n")
    log.write("---------------------------------\n")

# Scan all subfolders inside original/
all_folders = [os.path.join(root, d) for root, dirs, _ in os.walk(original_root) for d in dirs]

# Loop through each folder in original/
for original_folder in tqdm(all_folders, desc="Processing folders", unit="folder"):
    line_folder = os.path.join(line_root, os.path.relpath(original_folder, original_root))

    if not os.path.exists(line_folder):  # Skip if there is no matching folder in lineStrokes
        continue

    text_data = {}

    # Read all XML files inside the original/ folder
    xml_files = [f for f in os.listdir(original_folder) if f.endswith(".xml")]
    for filename in tqdm(xml_files, desc=f"Reading {os.path.basename(original_folder)}", leave=False, unit="file"):
        file_path = os.path.join(original_folder, filename)
        tree = ET.parse(file_path)
        root_xml = tree.getroot()

        # Extract all <TextLine> tags and store their id and text
        for textline in root_xml.findall(".//TextLine"):
            line_id = textline.get("id")
            line_text = textline.get("text")
            if line_id and line_text:
                text_data[line_id] = line_text

        # Extract writerID
        writer_id = root_xml.find(".//Form").get("writerID")

    # Modify files in the line/ folder
    for line_id, line_text in tqdm(text_data.items(), desc=f"Modifying {os.path.basename(original_folder)}", leave=False, unit="file"):
        line_file = os.path.join(line_folder, f"{line_id}.xml")

        if os.path.exists(line_file):
            tree = ET.parse(line_file)
            root_xml = tree.getroot()

            # Find <WhiteboardDescription> element
            wb_description = root_xml.find(".//WhiteboardDescription")
            if wb_description is not None:
                # Create Writer element
                writer = ET.Element("Writer")
                writer.set("writerID", writer_id)
                writer.tail = "\n  "

                # Create <Transcription> element
                transcription = ET.Element("Transcription")
                text_element = ET.SubElement(transcription, "Text")

                # Hardcoded formatting for proper indentation
                text_element.text = f"{line_text}"
                text_element.tail = "\n  "
                transcription.text = "\n    "
                transcription.tail = "\n  "

                # Insert after <WhiteboardDescription>
                parent = wb_description.getparent() if hasattr(wb_description, "getparent") else root_xml
                idx = list(parent).index(wb_description) + 1
                parent.insert(idx, writer)
                parent.insert(idx + 1, transcription)


                # Save to the output folder with the same structure
                output_folder = os.path.join(output_root, os.path.relpath(line_folder, line_root))
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f"{line_id}-labeled.xml")
                tree.write(output_path, encoding="utf-8", xml_declaration=True)

                # Log success
                with open(log_file, "a") as log:
                    log.write(f"Successfully modified: {line_id}.xml\n")
            else:
                # Log failure if <WhiteboardDescription> is missing
                with open(log_file, "a") as log:
                    log.write(f"Failed to modify (no <WhiteboardDescription>): {line_id}.xml\n")
        else:
            # Log failure if file is missing
            with open(log_file, "a") as log:
                log.write(f"Failed to modify (file not found): {line_id}.xml\n")

print(f"\nProcess completed! Log saved in '{log_file_name}'.")
