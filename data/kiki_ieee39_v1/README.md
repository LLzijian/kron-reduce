# Kiki IEEE39 primary real-data registry

Frozen on 2026-10-23 from the read-only `ieee39_postfault` and
`ieee39_expansion` source datasets. The raw 3.44 GB of HDF5 remains external;
set `KIKI_DATA_ROOT` to their parent directory. The local fallback is
`../kiki/private_data` relative to this project.

There are 591 accepted events: 497 stable and 94 unstable. Original assignment:
train 369, val 117, test 105. Development uses only original source train and
registered internal train/val groups. Original source keys such as `validation`
are retained byte-for-byte in copied source metadata.

The registry binds full source-file hashes, event IDs, group isolation, units,
physical-system identity and current generator/configuration snapshots.
Registration checks names, attributes and array shapes, without loading any
holdout trajectory arrays. Existing source QA supplies the full-array checks;
these are not represented as newly rerun numerical QA.

An external TASK3 exposure notice supersedes old claims that test was closed.
The old test pool is not fresh confirmation evidence. A new independent kiki
dataset and release registration are required before prospective test use.
This corpus uses one operating point and restored post-fault topology; it does
not supply multi-operating-point or Level 6 boundary coverage by itself.

Metadata, exposure evidence and checksums are versioned. Never edit source
trajectories, labels or splits. A new data revision gets a new registry.
