# ✂ CutOut Studio — rembg Backend

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> If you don't have a GPU, use `rembg` instead of `rembg[gpu]`

### 2. Run the server
```bash
python app.py
```
Server starts at **http://localhost:5000**

### 3. Open the website
Visit **http://localhost:5000** in your browser.

---

## Features
- 4 AI models: General, Portrait, Product, Anime
- Background color / transparent output
- Edge feathering
- Drop shadow with blur & opacity
- Output formats: PNG, JPG, WEBP
- Scale: 0.5x, 1x, 2x
- Before/after comparison slider
- History panel
- Copy to clipboard

## API Endpoints

### `GET /health`
Returns server status and available models.

### `POST /remove-background`
Form fields:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| image | file | required | Image file |
| model | string | general | general / portrait / product / anime |
| bg_color | string | "" | Hex color or empty for transparent |
| feather | int | 0 | Edge feather in px (0-20) |
| shadow | string | false | "true" / "false" |
| shadow_blur | int | 12 | Shadow blur radius |
| shadow_opacity | int | 60 | Shadow opacity 0-100 |
| format | string | png | png / jpg / webp |
| scale | float | 1.0 | Output scale factor |

Returns: Image file with headers:
- `X-Processing-Time` — ms taken
- `X-Image-Width` / `X-Image-Height` — output dimensions
