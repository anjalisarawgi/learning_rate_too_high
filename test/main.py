import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# === CONFIGURATION ===
IMG_DIR  = './'
XML_DIR  = './'
OUT_BASE = 'images'
JSON_PATH = './labels.json'

# === TEST MODE ===
TEST_MODE = True
SAMPLE_XML = 'P_0001_0001.xml'  # <-- replace with your test XML filename

# === SETUP ===
os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)

ns = {'a': 'http://www.loc.gov/standards/alto/ns-v4#'}
all_labels = {}

def find_image(basename):
    for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
        p = os.path.join(IMG_DIR, basename + ext)
        if os.path.exists(p):
            return p
    return None

# === FILE SELECTION ===
xml_files = [SAMPLE_XML] if TEST_MODE else os.listdir(XML_DIR)

for xml_fname in xml_files:
    if not xml_fname.lower().endswith('.xml'):
        continue

    basename = os.path.splitext(xml_fname)[0]
    xml_path = os.path.join(XML_DIR, xml_fname)
    img_path = find_image(basename)

    if img_path is None:
        print(f"Image not found for {basename}")
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = root.findall('.//a:TextLine', ns)

    out_dir = os.path.join(OUT_BASE, basename)
    os.makedirs(out_dir, exist_ok=True)
    page_labels = []

    print(f"Processing {basename}: {len(lines)} lines")

    for idx, line in enumerate(lines, start=1):
        lid = line.attrib.get('ID', f'line{idx}')
        poly_el = line.find('.//a:Polygon', ns)
        if poly_el is None: 
            continue

        coords = list(map(int, poly_el.attrib['POINTS'].split()))
        if len(coords) < 6:
            continue  # Skip malformed polygons
        pts = np.array(coords, dtype=np.int32).reshape(-1, 2)

        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        x, y, ww, hh = cv2.boundingRect(pts)

        if ww < 5 or hh < 5:
            continue

        crop_img = img[y:y+hh, x:x+ww]
        crop_mask = mask[y:y+hh, x:x+ww]
        out = np.full_like(crop_img, 255)
        out[crop_mask == 255] = crop_img[crop_mask == 255]

        out_path = os.path.join(out_dir, f"{lid}.png")
        cv2.imwrite(out_path, out)

        texts = [s.attrib.get('CONTENT', '') for s in line.findall('.//a:String', ns)]
        transcription = ' '.join(texts).strip()
        
        page_labels.append({
            "text": transcription,
            "image_path": os.path.join('test', 'images', basename, f"{lid}.png")
        })

        # Optional: visualize the crop for verification
        print(f"Line {lid} | Text: {transcription}")
        # cv2.imshow("Cropped Line", out)
        # cv2.waitKey(0)

    all_labels.setdefault("data", []).extend(page_labels)
    print(f"Saved {len(page_labels)} crops and labels for {basename}")

# Save to JSON (even if it's just one image)
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_labels["data"], f, ensure_ascii=False, indent=2)

print("Completed.")

# Optional: close any OpenCV windows
# cv2.destroyAllWindows()