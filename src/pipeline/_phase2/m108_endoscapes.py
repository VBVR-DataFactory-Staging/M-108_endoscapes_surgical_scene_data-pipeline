"""M-108: Endoscapes laparoscopic cholecystectomy anatomy segmentation.

Endoscapes2023 — CVS (Critical View of Safety) anatomical structures in laparoscopic
cholecystectomy. Pixel labels (1-6):
    1=cystic_plate, 2=hepato_cystic_triangle, 3=cystic_artery,
    4=cystic_duct, 5=gallbladder, 6=tool.

Layout:
    _extracted/M-108_Endoscapes/data/endoscapes/train/<frame>.jpg
                                        /semseg/<frame>.png  (uint8 class map)
Case D: single frame, loop 4s, multi-class overlay.
"""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np
from common import DATA_ROOT, write_task, COLORS, fit_square, overlay_multi

PID = "M-108"; TASK_NAME = "endoscapes_surgical_scene"; FPS = 8

CLASSES = [
    ("cystic_plate",            "yellow"),
    ("hepato_cystic_triangle",  "cyan"),
    ("cystic_artery",           "red"),
    ("cystic_duct",             "blue"),
    ("gallbladder",             "green"),
    ("tool",                    "magenta"),
]
COLOR_LIST = [(n, COLORS[c]) for n, c in CLASSES]

PROMPT = ("This is a laparoscopic cholecystectomy frame from the Endoscapes2023 dataset. "
          "Segment the 6 safety-critical anatomical structures for the Critical View of Safety: "
          "cystic plate (yellow), hepato-cystic triangle (cyan), cystic artery (red), "
          "cystic duct (blue), gallbladder (green), and surgical tool (magenta).")

def loop_frames(f, n): return [f.copy() for _ in range(n)]

def process_case(img_p, seg_p, idx):
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    seg = cv2.imread(str(seg_p), cv2.IMREAD_GRAYSCALE)
    if img is None or seg is None: return None
    if seg.shape[:2] != img.shape[:2]:
        seg = cv2.resize(seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    img_r = fit_square(img, 512)
    seg_r = cv2.resize(seg.astype(np.int16), (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    annot = overlay_multi(img_r, seg_r, COLOR_LIST)
    n = FPS * 4
    meta = {"task": "Endoscapes CVS anatomy segmentation", "dataset": "Endoscapes2023",
            "case_id": img_p.stem, "modality": "laparoscopic video frame",
            "classes": [n for n, _ in CLASSES], "colors": {n: c for n, c in CLASSES},
            "fps": FPS, "frames_per_video": n, "case_type": "D_single_image_loop"}
    return write_task(PID, TASK_NAME, idx, img_r, annot,
                      loop_frames(img_r, n), loop_frames(annot, n), loop_frames(annot, n),
                      PROMPT, meta, FPS)

def main():
    root = DATA_ROOT / "_extracted" / "M-108_Endoscapes" / "data" / "endoscapes"
    seg_dir = root / "semseg"
    pairs = []
    for split in ["train", "val", "test"]:
        img_dir = root / split
        if not img_dir.exists(): continue
        for img in sorted(img_dir.glob("*.jpg")):
            seg = seg_dir / f"{img.stem}.png"
            if seg.exists(): pairs.append((img, seg))
    print(f"  {len(pairs)} Endoscapes annotated frames")
    for i, (img, seg) in enumerate(pairs):
        d = process_case(img, seg, i)
        if d: print(f"  wrote {d}")

if __name__ == "__main__":
    main()
