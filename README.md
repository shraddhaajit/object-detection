# object-detection

YOLOv8s fine-tuned for 8-class urban scene detection (person, bicycle, car,
motorcycle, bus, truck, traffic light, stop sign) on COCO2017, with zero-shot
baseline comparison, held-out generalization testing, stratified train/val
splitting, and size-stratified recall analysis.

**Stack:** PyTorch, Ultralytics YOLOv8, COCO2017, Hugging Face Datasets
(streaming)

**Highlights:** balanced class sampling with starvation detection, disjoint
held-out test set for true generalization (not just in-pool validation),
zero-shot vs fine-tuned benchmarking, categorized failure analysis (false
negatives/positives/class confusion), deterministic training, experimental
pedestrian-proximity risk overlay.
