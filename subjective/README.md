

# Subjective Experiment (ACR 5-Point Scale)

## Files
| File | Purpose |
|---|---|
| `dashjs_sessions.json` | 120 sessions from the real dash.js player (4 methods × 30 test traces): bitrate level, startup delay, and stalls for each segment |
| `make_stimuli.py` | Stratified sampling with a fixed seed of 6 traces × 4 methods = 24 test videos + 2 practice videos, rendered according to the session logs (stall = frozen frame + loading icon) |
| `stimuli/` | Rendered videos `stim_XX.mp4` / `train_XX.mp4` (file names do not reveal the method), `stimuli.json` is the mapping table (do not open during the experiment), `playlist.json` is used by the rating program |
| `serve.py` + `app/index.html` | Rating program (local web page) |
| `analyze.py` | Data analysis |
| `experimenter_log.csv` | Experimenter log sheet (paper or electronic) |

## On the Day of the Experiment
1. Computer used for evaluation: Windows + Chrome, monitor set to native resolution (1920×1080), notifications and automatic updates disabled, screen brightness fixed.
2. In the `code` directory, run: `python subjective/serve.py`, then open `http://localhost:8000` in Chrome.
3. After the participant signs the informed consent form, the experimenter fills in the **participant ID** (S01, S02…, in order of arrival) and basic information (age group, gender, viewing frequency, vision) on the first page, clicks "Start", and hands the device to the participant.
4. The program automatically runs: instructions → 2 practice videos → 12 test videos → break (≥3 minutes) → 12 test videos → end. Each video is fully loaded into memory before playback, so no extra stalls are caused by disk reads; videos have no progress bar and cannot be seeked. The test order is randomized by participant ID, and two adjacent videos never come from the same trace.
5. After finishing, record one row in `experimenter_log.csv` (ID, date, start/end time, whether completed, abnormal situations). **Do not record names**; names appear only on the paper informed consent forms, which are kept separately by the person in charge.

## How Data Is Recorded
- Each rating is immediately appended to `subjective/data/ratings.csv` (one row per rating), so ratings already given are not lost even if the program is closed midway; at the end of each participant's session, a separate `subjective/data/Sxx.json` is saved.
- Columns of `ratings.csv`: subject, age_group, gender, viewing_freq, vision, phase(train/test), trial, file, rating(1–5), rt_s (reaction time), played_s, complete, time.
- Midway withdrawal: keep the existing data in `ratings.csv`; if the participant requests deletion, delete all rows for that ID and the `Sxx.json` file.
- After each day's experiment, back up `subjective/data/` to the encrypted lab computer.

## Analysis
```
python subjective/analyze.py --ratings subjective/data/ratings.csv --stimuli subjective/stimuli/stimuli.json
```
Output: outlier participants removed according to ITU-T P.913 (correlation coefficient with the average scores of others < 0.75), MOS and 95% confidence intervals for each method (all / 3G and broadband / 4G/5G), paired differences between NeSyStream and each method (Wilcoxon test + bootstrap confidence intervals), and the correlation between P.1203 and MOS. Results are written to `subjective/data/results.json`.

## Re-rendering
```
python subjective/make_stimuli.py --sessions subjective/dashjs_sessions.json --src_dir deploy_dashjs/src --out subjective/stimuli
```
Requires the 13-level 60-second source files used in the dash.js experiment (`deploy_dashjs/src/level_XX.mp4`) and ffmpeg.
