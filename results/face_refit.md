# Face model ELM re-fit (FER-2013)

Produced by `scripts/refit_face_elm.py`. Only the ELM output layer B was re-solved;
the ResNet, the ViT file and the ELM hidden layer (W, b) are unchanged.
Full FER-2013 splits.

- Chosen ridge strength (by validation AUC): lambda = 100.0
- Validation AUC: 0.7104
- **Test AUC: 0.7025** (pass bar >= 0.70: PASS)
- Test accuracy (calibrated, 0.5 cut): 0.6492
- Test Brier (calibrated): 0.2193
- Always-guess-majority accuracy: 0.5341

New files: artifacts/face/refit/elm_B.npy, scaler_mean.npy, scaler_std.npy, platt.json.
The raw ELM output is mapped to a probability by a logistic fitted on the validation split.
