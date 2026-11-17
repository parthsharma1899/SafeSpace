# Virtual-subject validation

Produced by `scripts/run_virtual_subjects.py`. Each simulated person combines
real, held-out model scores from different real people, drawn to match a
simulated stress curve (seeds 1000-1199).

Each simulated person combines scores from different real people. This assumes
the modalities are independent once the stress state is known. Earlier work on
such combined ('chimeric') subjects in biometrics found that results do not
always carry over to real people, so these numbers are supporting evidence only.

Score pools (stressed / not), real held-out scores per modality: wrist 311/670, text 369/346, voice 768/672, face 506/450.

| method | MAE | bias | episode AUC | false-alarm share |
|---|---:|---:|---:|---:|
| v1 | 0.2301 | +0.1870 | 0.8874 | 0.1624 |
| mean | 0.1624 | -0.0305 | 0.9535 | 0.0101 |
| median | 0.1722 | -0.0335 | 0.9530 | 0.0110 |
| mean+ema | 0.1257 | -0.0312 | 0.9922 | 0.0002 |
| median+ema | 0.1291 | -0.0342 | 0.9920 | 0.0002 |
| v2 | 0.1252 | -0.0265 | 0.9830 | 0.0019 |
| v3 | 0.1384 | -0.0099 | 0.9878 | 0.0005 |
