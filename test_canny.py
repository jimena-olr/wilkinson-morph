from pathlib import Path
import time
import cv2
from dl_morph_labelling.preprocessing import canny_vial_crop

# adjust to an existing C3 folder on your repo
src = Path("dl_morph_labelling/C3/other")
out = Path("dl_morph_labelling/images/preprocessed/noncrystal_debug")
out.mkdir(parents=True, exist_ok=True)

imgs = sorted([p for p in src.glob("*") if p.suffix.lower() in (".png",".jpg",".jpeg")])
if not imgs:
    raise SystemExit(f"No images found in {src}")
p = imgs[0]
img = cv2.imread(str(p))
t0 = time.time()
cropped = canny_vial_crop(img)
dt = time.time() - t0
print(f"Processed {p.name} in {dt:.3f}s, result type: {type(cropped)}, shape: {getattr(cropped,'shape',None)}")
cv2.imwrite(str(out / ("debug_" + p.name)), cropped)
print("Wrote:", out / ("debug_" + p.name))