import os

# Force CPU — disables CUDA so no GPU errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"

from flask import Flask, request, render_template, send_file, jsonify
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image, ImageFilter
import onnxruntime as ort
import io
import time

app = Flask(__name__)
CORS(app)

# Available models
MODELS = {
    "general":  "u2net",             # Best all-purpose
    "portrait": "u2net_human_seg",   # People & faces
    "anime":    "isnet-anime",       # Anime / illustrations
    "product":  "silueta",           # Objects & products
}

# CPU-only session options
SESS_OPTS = ort.SessionOptions()
SESS_OPTS.inter_op_num_threads = 4
SESS_OPTS.intra_op_num_threads = 4
CPU_PROVIDERS = ["CPUExecutionProvider"]

# Cache loaded sessions so model is loaded only once
_sessions = {}


def get_session(model_name):
    """Return a cached CPU session for the given model."""
    if model_name not in _sessions:
        print(f"[CutOut] Loading model: {model_name} (CPU)")
        _sessions[model_name] = new_session(
            model_name,
            sess_options=SESS_OPTS,
            providers=CPU_PROVIDERS,
        )
        print(f"[CutOut] Model ready: {model_name}")
    return _sessions[model_name]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": list(MODELS.keys()),
        "version": "2.1",
        "device": "CPU",
    })


@app.route("/remove-background", methods=["POST"])
def remove_background():
    start = time.time()

    # Validate input
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # Options
    model_key   = request.form.get("model", "general")
    bg_color    = request.form.get("bg_color", "").strip()
    feather     = int(request.form.get("feather", 0))
    shadow      = request.form.get("shadow", "false") == "true"
    shadow_blur = int(request.form.get("shadow_blur", 12))
    shadow_opac = int(request.form.get("shadow_opacity", 60))
    out_format  = request.form.get("format", "png").lower()
    scale       = float(request.form.get("scale", 1.0))

    model_name = MODELS.get(model_key, "u2net")

    # Remove background
    try:
        input_bytes  = file.read()
        session      = get_session(model_name)
        output_bytes = remove(input_bytes, session=session)
    except Exception as e:
        print(f"[CutOut] rembg error: {e}")
        return jsonify({"error": f"rembg failed: {str(e)}"}), 500

    img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # Feather edges
    if feather > 0:
        r, g, b, a = img.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=feather))
        img = Image.merge("RGBA", (r, g, b, a))

    # Scale
    if scale != 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img   = img.resize((new_w, new_h), Image.LANCZOS)

    # Compose background + shadow
    if bg_color or shadow:
        canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))

        if shadow:
            _, _, _, mask = img.split()
            shadow_rgba   = (0, 0, 0, int(shadow_opac * 2.55))
            shadow_img    = Image.new("RGBA", img.size, shadow_rgba)
            shadow_img.putalpha(mask)
            shadow_img    = shadow_img.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
            shadow_canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
            shadow_canvas.paste(shadow_img, (6, 8))
            canvas        = Image.alpha_composite(canvas, shadow_canvas)

        if bg_color.startswith("#") and len(bg_color) >= 7:
            try:
                hex_col = bg_color.lstrip("#")
                rv = int(hex_col[0:2], 16)
                gv = int(hex_col[2:4], 16)
                bv = int(hex_col[4:6], 16)
                bg_layer = Image.new("RGBA", img.size, (rv, gv, bv, 255))
                canvas   = Image.alpha_composite(bg_layer, canvas)
            except ValueError:
                pass

        canvas = Image.alpha_composite(canvas, img)
        img    = canvas

    # Encode output
    img_io  = io.BytesIO()
    elapsed = round((time.time() - start) * 1000)

    if out_format == "jpg":
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.split()[3])
        flat.save(img_io, "JPEG", quality=95)
        mime  = "image/jpeg"
        fname = "cutout.jpg"
    elif out_format == "webp":
        img.save(img_io, "WEBP", quality=95)
        mime  = "image/webp"
        fname = "cutout.webp"
    else:
        img.save(img_io, "PNG")
        mime  = "image/png"
        fname = "cutout.png"

    img_io.seek(0)
    print(f"[CutOut] Done in {elapsed}ms  model={model_name}  size={img.width}x{img.height}")

    resp = send_file(img_io, mimetype=mime, as_attachment=False, download_name=fname)
    resp.headers["X-Processing-Time"] = str(elapsed)
    resp.headers["X-Image-Width"]     = str(img.width)
    resp.headers["X-Image-Height"]    = str(img.height)
    return resp


# Pre-load default model at startup so first request isn't slow
with app.app_context():
    try:
        print("[CutOut] Pre-loading u2net model...")
        get_session("u2net")
        print("[CutOut] Model ready!")
    except Exception as e:
        print(f"[CutOut] Model preload failed: {e}")


if __name__ == "__main__":
    print("[CutOut] Starting server on http://localhost:5000")
    app.run(debug=True, port=5000)