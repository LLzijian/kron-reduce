# TEST exposure notice for subsequent work

On2026-09-27 this TASK3 release consumed the accepted fresh expansion+holdout-supplement TEST after explicit user authorization and endpoint freeze. All76 eligible event arrays were read once into a durable TASK3 cache;10 latency-excluded event payloads were not opened. Old Pilot TEST remains excluded and was not reread.

Authoritative evidence: `runs/campaign2_final/holdout_access.json` and `freeze_manifest.json`. The pre-read KIKI current_state snapshot said TEST closed; that historical statement is no longer a fresh-holdout guarantee after this release. KIKI was kept read-only, so its state file has not been updated here. Any later TASK2 models, including0102, must disclose this existing exposure and must not present these events as a newly unseen confirmation set. No new training permission follows from this notice.
