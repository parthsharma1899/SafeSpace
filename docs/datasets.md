# Datasets

One section per dataset: where it comes from, what SafeSpace uses from it, and
its licence. Each section is filled in when that dataset is first downloaded.
Datasets live under `data/` and are never committed. Check a dataset's licence
here before publishing any weight trained on it.

## WESAD (wrist model)

- Source: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download (the official
  download linked from https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/).
- Used: the wrist (Empatica E4) BVP, EDA and 3-axis ACC signals, and the labels
  (1 baseline, 2 stress, 3 amusement), from all 15 subjects (S2-S17; S1 and S12
  were discarded by the dataset authors for sensor malfunction). Temperature is
  not used. The chest (RespiBAN) signals are not used.
- Licence/terms (from `wesad_readme.pdf`): research dataset; "If you publish
  material based on this dataset, please reference the publication" (Schmidt
  et al., ICMI 2018). Contact: Philip Schmidt and Attila Reiss (Bosch). A model
  trained on WESAD is published here only if these terms allow it.

## Dreaddit (text model)

- Source: http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip (Turcan &
  McKeown, LOUHI 2019).
- Used: the `text` and `label` columns of `dreaddit-test.csv` (715 rows) to
  measure the existing text model.
- Licence: none stated (checked 2026-11-17: not in the zip, the paper or the
  ACL Anthology page). The posts come from Reddit, so Reddit's content terms
  also apply. Ask the authors before publishing weights trained on it.

## RAVDESS speech (voice model)

- Source: https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip
  (Livingstone & Russo, 2018).
- Used: all 1,440 speech clips to measure the existing voice model. Emotions
  04, 05, 06, 07 (sad, angry, fearful, disgust) are stressed; 01, 02, 03, 08
  are relaxed.
- Licence: CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike), per the
  Zenodo record and its library listings (checked 2026-11-17). Commercial
  licences are sold by the authors (ravdess@gmail.com).

## FER-2013 (face model)

- Source: https://huggingface.co/datasets/chitradrishti/fer2013 (a mirror of the
  Kaggle "Challenges in Representation Learning" FER-2013 set).
- Used: the test split to record the face model's behaviour. Emotions 0, 1, 2, 4
  (angry, disgust, fear, sad) are stressed; 3, 5, 6 (happy, surprise, neutral)
  are relaxed, following the paper's mapping.
- Licence: no clear licence (checked 2026-11-17). Access is through the Kaggle
  competition rules, and projects built on it do not re-host the images. A
  mirror's "MIT" tag is the mirror's, not the dataset's. Check the Kaggle rules
  before publishing the re-fitted face weights.

## UBFC-Phys (real-person validation, Phase 4)

- Source: IEEE DataPort, DOI 10.21227/5da0-7344 (login required). The owner
  registers and downloads into `data/raw/ubfc_phys/`.
- Used: face video and wrist BVP/EDA for rest (T1) against speech (T2) and
  arithmetic (T3).
- Licence: per the IEEE DataPort terms accepted at download.

## StressID (real-person validation, Phase 4)

- Source: https://project.inria.fr/stressid (licence signed by a permanent
  academic, emailed to Inria).
- Used: face video, voice and physiological signals with StressID's own stress
  labels.
- Licence: StressID non-commercial research licence; models trained on it may
  not be redistributed.
