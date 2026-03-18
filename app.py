import os
import sys
import threading
import warnings
warnings.filterwarnings("ignore")

# Must be FIRST before any other imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["ONNXRUNTIME_PROVIDERS"] = "CPUExecutionProvider"

# Redirect stderr temporarily to suppress GPU discovery errors on import
import io as _io
_old_stderr = sys.stderr
sys.stderr = _io.StringIO()

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
finally:
    sys.stderr = _old_stderr

from flask import Flask, request, render_template, send_file, jsonify
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image, ImageFilter
import io
import time

app = Flask(__name__)
CORS(app)

MODELS = {
    "general":  "u2net",
    "portrait": "u2net_human_seg",
    "anime":    "isnet-anime",
    "product":  "silueta",
}

SESS_OPTS = ort.SessionOptions()
SESS_OPTS.inter_op_num_threads = 2
SESS_OPTS.intra_op_num_threads = 2
SESS_OPTS.log_severity_level = 3
CPU_PROVIDERS = ["CPUExecutionProvider"]

_sessions = {}
_lock = threading.Lock()


def get_session(model_name):
    if model_name not in _sessions:
        with _lock:
            if model_name not in _sessions:
                print(f"[CutOut] Loading: {model_name}", flush=True)
                try:
                    _sessions[model_name] = new_session(
                        model_name,
                        sess_options=SESS_OPTS,
                        providers=CPU_PROVIDERS,
                    )
                    print(f"[CutOut] Ready: {model_name}", flush=True)
                except Exception as e:
                    print(f"[CutOut] Failed {model_name}: {e}", flush=True)
                    if model_name != "u2net":
                        return get_session("u2net")
                    raise e
    return _sessions[model_name]


def preload_models():
    time.sleep(3)
    print("[CutOut] Preloading models...", flush=True)
    for key, model_name in MODELS.items():
        try:
            get_session(model_name)
        except Exception as e:
            print(f"[CutOut] Skipping {key}: {e}", flush=True)
    print("[CutOut] All models ready!", flush=True)


threading.Thread(target=preload_models, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ping")
def ping():
    return "pong", 200


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "loaded": list(_sessions.keys()),
        "version": "2.5",
    })


@app.route("/remove-background", methods=["POST"])
def remove_background():
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    model_key   = request.form.get("model", "general")
    bg_color    = request.form.get("bg_color", "").strip()
    feather     = int(request.form.get("feather", 0))
    shadow      = request.form.get("shadow", "false") == "true"
    shadow_blur = int(request.form.get("shadow_blur", 12))
    shadow_opac = int(request.form.get("shadow_opacity", 60))
    out_format  = request.form.get("format", "png").lower()
    scale       = float(request.form.get("scale", 1.0))

    model_name = MODELS.get(model_key, "u2net")

    try:
        input_bytes  = file.read()
        session      = get_session(model_name)
        output_bytes = remove(input_bytes, session=session)
    except Exception as e:
        print(f"[CutOut] Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

    img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    if feather > 0:
        r, g, b, a = img.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=feather))
        img = Image.merge("RGBA", (r, g, b, a))

    if scale != 1.0:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS
        )

    if bg_color or shadow:
        canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
        if shadow:
            _, _, _, mask = img.split()
            sh = Image.new("RGBA", img.size, (0, 0, 0, int(shadow_opac * 2.55)))
            sh.putalpha(mask)
            sh = sh.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
            sc = Image.new("RGBA", img.size, (0, 0, 0, 0))
            sc.paste(sh, (6, 8))
            canvas = Image.alpha_composite(canvas, sc)
        if bg_color.startswith("#") and len(bg_color) >= 7:
            try:
                h = bg_color.lstrip("#")
                bl = Image.new("RGBA", img.size, (
                    int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255
                ))
                canvas = Image.alpha_composite(bl, canvas)
            except ValueError:
                pass
        img = Image.alpha_composite(canvas, img)

    buf     = io.BytesIO()
    elapsed = round((time.time() - start) * 1000)

    if out_format == "jpg":
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.split()[3])
        flat.save(buf, "JPEG", quality=95)
        mime, fname = "image/jpeg", "cutout.jpg"
    elif out_format == "webp":
        img.save(buf, "WEBP", quality=95)
        mime, fname = "image/webp", "cutout.webp"
    else:
        img.save(buf, "PNG")
        mime, fname = "image/png", "cutout.png"

    buf.seek(0)
    print(f"[CutOut] {elapsed}ms {model_name} {img.width}x{img.height}", flush=True)

    resp = send_file(buf, mimetype=mime, as_attachment=False, download_name=fname)
    resp.headers["X-Processing-Time"] = str(elapsed)
    resp.headers["X-Image-Width"]     = str(img.width)
    resp.headers["X-Image-Height"]    = str(img.height)
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[CutOut] Starting on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
