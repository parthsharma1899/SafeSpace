# Face model: compact ELM file

Produced by `scripts/compact_face_elm.py`. The ELM hidden layer W is stored in
float16 instead of float64; nothing is retrained.

- Size: 2.45 GB -> 0.61 GB
- FER-2013 test images compared: 416 (60 per emotion)
- Largest score difference: 0.00020
- AUC with the original file: 0.6904
- AUC with the compact file: 0.6904
