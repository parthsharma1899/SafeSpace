---
title: SafeSpace
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: A one-minute stress check-in that learns your normal
---

# SafeSpace

A one-minute stress check-in. Write how you feel, read one sentence aloud and
take a selfie; four models score stress from your words, voice and expression,
and a fusion method combines them. With an account, SafeSpace learns your
normal over several check-ins and compares each result with it.

**Use the app in its own tab: <https://parth18s-safespace.hf.space>** (the camera,
microphone and Google sign-in do not work inside the frame on this page).

- **Research prototype, not a medical device.** It can be wrong; do not use it
  to make decisions about your health. If you feel unsafe or in crisis, contact
  your local emergency number or a crisis line.
- **Privacy.** Your words, recording and photo are scored in memory and deleted
  straight away. If you sign in and agree, only the scores (one number per
  signal) and times are saved, and you can delete them from your account page.
- The app sleeps after a period without visitors; the first visit after that
  takes a minute or two while the models load.

Code, model cards and results:
[github.com/parthsharma1899/SafeSpace](https://github.com/parthsharma1899/SafeSpace).
