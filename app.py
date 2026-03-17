import os
import threading

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

MODELS = {
    "general":  "u2net",
    "portrait": "u2net_human_seg",
    "anime":    "isnet-anime",
    "product":  "silueta",
}

SESS_OPTS = ort.SessionOptions()
SESS_OPTS.inter_op_num_threads = 4
SESS_OPTS.intra_op_num_threads = 4
CPU_PROVIDERS = ["CPUExecutionProvider"]

_sessions = {}
_loading_lock = threading.Lock()


def get_session(model_name):
    if model_name not in _sessions:
        with _loading_lock:
            if model_name not in _sessions:
                print(f"[CutOut] Loading model: {model_name}")
                try:
                    _sessions[model_name] = new_session(
                        model_name,
                        sess_options=SESS_OPTS,
                        providers=CPU_PROVIDERS,
                    )
                    print(f"[CutOut] Model ready: {model_name}")
                except Exception as e:
                    print(f"[CutOut] Failed {model_name}: {e}, falling back to u2net")
                    if model_name != "u2net":
                        return get_session("u2net")
                    raise e
    return _sessions[model_name]


def preload_models():
    """Load all models in background thread — won't block server startup."""
    print("[CutOut] Background model preload started...")
    for key, model_name in MODELS.items():
        try:
            get_session(model_name)
        except Exception as e:
            print(f"[CutOut] Could not preload {key}: {e}")
    print("[CutOut] All models loaded!")


# Start preloading in background — server starts immediately
threading.Thread(target=preload_models, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": list(MODELS.keys()),
        "loaded_models": list(_sessions.keys()),
        "version": "2.3",
        "device": "CPU",
    })


@app.route("/remove-background", methods=["POST"])
def remove_background():
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["image"]
    if file.filename == "":
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
        print(f"[CutOut] rembg error: {e}")
        return jsonify({"error": f"rembg failed: {str(e)}"}), 500

    img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    if feather > 0:
        r, g, b, a = img.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=feather))
        img = Image.merge("RGBA", (r, g, b, a))

    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    if bg_color or shadow:
        canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))

        if shadow:
            _, _, _, mask = img.split()
            shadow_img = Image.new("RGBA", img.size, (0, 0, 0, int(shadow_opac * 2.55)))
            shadow_img.putalpha(mask)
            shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
            shadow_canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
            shadow_canvas.paste(shadow_img, (6, 8))
            canvas = Image.alpha_composite(canvas, shadow_canvas)

        if bg_color.startswith("#") and len(bg_color) >= 7:
            try:
                h = bg_color.lstrip("#")
                bg_layer = Image.new("RGBA", img.size, (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16), 255))
                canvas = Image.alpha_composite(bg_layer, canvas)
            except ValueError:
                pass

        img = Image.alpha_composite(canvas, img)

    img_io  = io.BytesIO()
    elapsed = round((time.time() - start) * 1000)

    if out_format == "jpg":
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.split()[3])
        flat.save(img_io, "JPEG", quality=95)
        mime, fname = "image/jpeg", "cutout.jpg"
    elif out_format == "webp":
        img.save(img_io, "WEBP", quality=95)
        mime, fname = "image/webp", "cutout.webp"
    else:
        img.save(img_io, "PNG")
        mime, fname = "image/png", "cutout.png"

    img_io.seek(0)
    print(f"[CutOut] Done in {elapsed}ms  model={model_name}  size={img.width}x{img.height}")

    resp = send_file(img_io, mimetype=mime, as_attachment=False, download_name=fname)
    resp.headers["X-Processing-Time"] = str(elapsed)
    resp.headers["X-Image-Width"]     = str(img.width)
    resp.headers["X-Image-Height"]    = str(img.height)
    return resp


if __name__ == "__main__":
    print("[CutOut] Starting server on http://localhost:5000")
    app.run(debug=True, port=5000)