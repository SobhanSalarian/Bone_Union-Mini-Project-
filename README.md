# Mandibular Bone Union Detection — Project Report

## Dataset Overview

Two datasets are used, one per part.

**Dataset 1 (Part 1).** A single full CT scan of one patient, provided as 321
stacked cross-sectional images (`ct_sample/DICOM/`). Used to get familiar with how
CT data is structured before building anything.

**Dataset 2 (Part 2).** 820 individual CT slice images (`dataset/images/`), each
with a matching label file (`dataset/labels/`) marking where the "osteotomy
sites" — the graft-to-jaw joins — appear, as bounding boxes. These slices come from
85 different patients across 160 scanning sessions; only slices that already show a
join were kept. This is the dataset the detection model is trained and tested on.

---

## Part 1 — Understanding the CT Data

### 1.1 Anatomical Viewing Planes

A CT scan is a stack of thin cross-sectional images. That stack can be viewed from
three standard directions:

**Axial plane** — a horizontal cut, separating top from bottom. Like slicing a loaf
of bread into round slices. This is the view the provided data uses, and the
standard view for inspecting the jaw, since one slice shows the whole arch at once.

![Axial plane](images/axial_plane.png)

**Coronal plane** — a vertical cut, separating front from back. Like a slice through
the face from ear to ear.

![Coronal plane](images/coronal_plane.png)

**Sagittal plane** — a vertical cut, separating left from right. Gives a "profile"
view of the jaw.

![Sagittal plane](images/sagittal_plane.png)

The three are just three different directions to slice the same 3D data — no
information is added or lost by choosing one over another.

### 1.2 Looking at the Sample CT Volume

*(Code in `notebooks/part1_ct_analysis.ipynb`.)*

Loading the scan gives a 3D array of shape **(321, 512, 512)** — 321 slices, each
512×512 pixels.

Checking the brightness range turned up something worth knowing: values ran from
**-16,040 to 32,767**, far outside what real tissue should show (normally about
-1,000 for air up to a few thousand for dense bone). Neither extreme is real
anatomy — 32,767 is the maximum a pixel can technically hold, caused by metal
hardware "blowing out" the scanner's range, and -16,040 is a padding value the
scanner writes outside the area it actually scanned. Any real use of this data
needs to clip/adjust the brightness range first rather than trust the raw min/max —
which is what was done to produce readable pictures below.

| Slice 0 | Slice 80 |
|---|---|
| ![slice 0](images/sample_slices/slice_000.jpg) | ![slice 80](images/sample_slices/slice_080.jpg) |

| Slice 160 | Slice 240 | Slice 320 |
|---|---|---|
| ![slice 160](images/sample_slices/slice_160.jpg) | ![slice 240](images/sample_slices/slice_240.jpg) | ![slice 320](images/sample_slices/slice_320.jpg) |

---

## Part 2 — Osteotomy Site Detection Pipeline

### The Dataset

*(Code in `notebooks/part2_dataset_inspection.ipynb`.)*

820 images, 820 matching labels, no orphans. They come from 85 patients across 160
scanning sessions. One important thing stood out: the images are **not evenly
spread across patients** — just 5 of the 85 patients supply 44% of all images. That
had to be accounted for when splitting the data (below). Almost every image has at
least one labelled box (1,109 boxes total across 820 images), since the slices were
already pre-selected to show a join.

### Splitting the Data

*(Code in `notebooks/part2_dataset_inspection.ipynb`.)*

The data is split **by patient**, not by image — every image from a given patient
goes entirely into one of train/validation/test. Splitting by image instead would
let near-identical slices of the same patient leak across the split, letting the
model partly "recognise" a patient instead of genuinely learning to find joins.

Because a handful of patients contribute far more images than others, patients were
assigned to splits with a balancing rule rather than pure randomness, so no single
patient could dominate the validation or test set. Result:

```
train: 35 patients, 574 images (70%)
val:   25 patients, 123 images (15%)
test:  25 patients, 123 images (15%)
```

### Building the Model — Three Attempts

*(Code in `notebooks/part2_train.ipynb`, `part2_train_cropped.ipynb`,
`part2_train_metalmask.ipynb`.)*

**Framework.** YOLOv8, a standard, well-tested object detector, starting from a
pretrained checkpoint. The labels were already in the right format for it, and it
works well on small datasets like this one via transfer learning.

**Constraint.** This machine has no GPU. A full training run at the images' native
size would have taken 3-4 hours per attempt — too slow to try more than one idea.
Shrinking the images and caching them in memory cut that to under an hour, making
real experimentation possible.

**Approach 1 — baseline.** Resize each image down and train. Result: the model
finds about half the real joins (recall 0.55), and **43% of everything it flags
turns out to be wrong** (precision 0.56). Looking at *why* it was wrong: it kept
mistaking other metal screws and plates elsewhere in the image for real joins.

> **This — telling a real join apart from a nearby screw — turned out to be the
> project's central, unsolved problem, and the rest of the work focused on it.**

**Approach 2 — crop instead of shrink.** Shrinking a photo throws away detail, and
the joins are always roughly in the same part of the image, so instead of shrinking
the whole photo, a fixed window around that area was cropped out at full detail
instead — same training speed, but no lost sharpness. Result: mixed. Boxes were
drawn more precisely once found, but the screw-vs-join confusion didn't improve —
if anything, the false-positive rate went up slightly (47%).

**Approach 3 — tell the model where the metal is.** Screws are, by a wide margin,
the single brightest thing in any of these images. Rather than make the model
figure that out for itself from brightness alone, it was handed that fact directly:
alongside the normal image, a second, pre-computed map marking "this is definitely
metal" was given as an extra input. That freed the model to spend its effort on the
harder part — learning what a real join actually looks like — instead of
re-discovering "very bright = metal" on its own. **This worked.**

**Result on the held-out test set** (123 images, 25 patients, never used in
training):

| | Approach 1 (resize) | Approach 2 (crop) | **Approach 3 (metal mask)** |
|---|---|---|---|
| Precision | 0.56 | 0.59 | **0.66** |
| Recall | 0.55 | 0.53 | 0.54 |
| False positive rate | 43% | 47% | **30%** |
| Fully correct images | 32% | 29% | **41%** |

Approach 3 cut the false-positive rate by nearly a third while barely costing any
recall — a real improvement, not a trade-off. Looking again at what it still gets
wrong: the remaining mistakes are now mostly near-misses right next to a real join,
rather than confidently flagging a completely unrelated screw. The core problem
shrank; it didn't disappear.

---

## Discussion

### Limitations

- **No GPU.** All training ran on CPU, which limited both image resolution and how
  many ideas could realistically be tried in the time available.
- **Telling a real join apart from a nearby screw remains only partly solved.**
  Approach 3 cut false positives from 43% to 30%, but 30% is still a lot of wrong
  answers a doctor would have to double-check.
- **Small, unevenly distributed dataset.** 85 patients isn't a lot to train a
  detector on, and 5 of them supply nearly half the images — the model has seen a
  narrower range of real anatomy than "820 images" suggests.
- **Each slice is judged alone.** A real join actually spans several neighbouring
  slices, but the model has no way to use that — it makes an independent guess on
  every single slice.

### Improvements

- **The main lesson from this project: give the model domain knowledge directly,
  rather than hoping it discovers it, or trying to correct it afterward.** The
  metal-mask idea (Approach 3) worked because it did exactly that, and did better
  than a cropping change (Approach 2) that only helped indirectly. The clearest next
  step is combining Approach 3's metal mask with Approach 2's sharper crop, since
  they solve two different parts of the problem.
- **A GPU would remove the resolution/speed trade-off entirely** — training at full
  image size, or for longer, without the CPU time constraint.
- **More patients, not just more images from the same ones**, would likely improve
  how well the model generalises, given how few patients the data currently spans.
- **Testing across more than one train/test split** (not just the single fixed one
  used here) would give a more trustworthy read on real performance, since the
  dataset is small enough that one split's numbers can be noisy.

---

## Short Survey

- How interested are you in computer vision? (0-5): 3
- How interested are you in medical image processing? (0-5): 4
- How interested were you in this project? (0-5): 5
