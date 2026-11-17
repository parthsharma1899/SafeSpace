# Owner decisions (recorded as they are given, with dates)

- 2026-11-15, step 0.5 Q1 (commit per phase): YES. Commit finished phases to
  local `revamp`; no push until a later approval gate.
- 2026-11-15, step 0.5 Q2 (delete .git garbage): YES (after confirming safety).
  Owner asked whether it was safe; verified the two tmp_pack_* files are
  orphaned (git fsck clean, commits intact) and confirmed deletion is safe.
- 2026-11-15, step 0.5 Q3 (rewrite public history to remove 5 personal media
  files): YES to prepare the rewrite; a second yes is still required before
  the force-push itself (step 0.7).

- 2026-11-15, step 3D.3 (face model): RE-FIT the ELM's last layer on FER-2013,
  keeping ResNet, the ViT file and the ELM hidden layer unchanged. Writes new
  files under artifacts/face/refit/, touches no original. Ships ON only if test
  AUC >= 0.70, else stays OFF; number reported either way.

- 2026-11-17, web app direction: the public web app is a finished product people
  can use, not a research demo. No v1-versus-mean comparison in the app; fusion
  comparisons stay in the docs and results. The playground and the TypeScript
  fusion ports (PLAN 6.2-6.3) were removed accordingly.
