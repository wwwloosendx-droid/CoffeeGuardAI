# CoffeeGuardAI Checkpoint

Start from here next time.

## Current Status

- `python app.py` now runs using the project virtual environment.
- Dashboard registration crash was fixed.
- Dashboard now receives `chart_labels` and `chart_values`.
- App paths now use absolute paths based on `app.py`, so files are found even if Python starts from another folder.
- Upload and logout routes were added.
- Register password confirmation is checked.
- Register page now shows errors.
- Upload page was redesigned.
- Dashboard was redesigned.
- Prediction now saves:
  - disease result
  - confidence
  - uploaded image
  - AI heatmap
  - yield estimate
  - yield message
- Heatmaps are generated into `static/heatmaps`.
- Old predictions may not have heatmaps, but new uploads will.

## Main Files Changed

- `app.py`
- `templates/dashboard.html`
- `templates/upload.html`
- `templates/register.html`

## Run Command

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000/login
```

## Next Improvements

- Improve disease/yield formula using real agronomy data.
- Add a result page after upload instead of only redirecting to dashboard.
- Add delete/download report buttons.
- Add admin dashboard if needed.
