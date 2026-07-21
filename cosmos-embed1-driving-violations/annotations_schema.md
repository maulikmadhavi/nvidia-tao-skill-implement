# Input contract: annotations.csv

One row per violation event. Place at `workspace\data\annotations.csv` (00_chunk_videos
copies it there in the self-test; do the same for real data — scripts read it from `/data`).

```csv
video,class,start_sec,end_sec
lesson_A01.mp4,trainer_phone_use,83,97
lesson_A01.mp4,driver_hands_away,145,151
lesson_B07.mp4,trainer_no_seatbelt,0,278
```

| column | type | rules |
|---|---|---|
| `video` | str | source video filename (extension optional; matched by stem against the files in the videos dir) |
| `class` | str | one of the ids in `scripts\taxonomy.py` (NOT `no_violation` — absence of events = negative) |
| `start_sec` | int | event start, seconds from video start |
| `end_sec` | int | event end (exclusive-ish; must be > start_sec) |

Notes
- Overlapping events of different classes on the same video are fine (that is the
  multi-label case the pipeline is built for).
- Events shorter than 0.5s never overlap a chunk by >= 0.5s and are dropped with a
  warning by `00_chunk_videos.py` — extend them to >= 1s when annotating.
- Videos with NO rows are treated as all-negative (pure no-violation lessons) — include
  them; they are the realistic negatives.
- Different column names? Pass `--columns video=<v>,class=<c>,start=<s>,end=<e>` to
  `00_chunk_videos.py`.
- Class list changes (drop `driver_sleeping`, add a class): edit `scripts\taxonomy.py`
  ONLY — captions/labels propagate everywhere from there. Keep captions concrete and
  role-specific ("the instructor in the passenger seat ...").
