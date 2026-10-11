"""Konverter gambar (jpg/png) -> model 3D .glb berwarna. Berjalan lokal, gratis."""
import io
import os
import sys
import threading
import time
import types
import gc
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage as ndi
from scipy import sparse
from transformers import pipeline

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"  # Apache-2.0
MAX_BYTES = 15 * 1024 * 1024
TRIPOSR_DIR = Path(__file__).parent / "TripoSR"
state: dict = {}
gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    state["device"] = device
    print(f"[init] Memuat model depth ({MODEL_ID}) di {device}...")
    state["depth"] = pipeline("depth-estimation", model=MODEL_ID, device=device)
    print("[init] Siap: buka http://127.0.0.1:8000")
    yield
    state.clear()


app = FastAPI(title="Image to GLB", lifespan=lifespan)


def to_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert("RGB")


# ==================== MODE RELIEF ====================

def build_relief(img: Image.Image, relief: float, detail: int) -> bytes:
    t0 = time.time()
    img = to_rgb(img)
    img.thumbnail((1536, 1536), Image.LANCZOS)
    w0, h0 = img.size

    depth = state["depth"](img)["depth"].convert("L")

    scale = detail / max(w0, h0)
    cols = max(2, round(w0 * scale))
    rows = max(2, round(h0 * scale))

    d = depth.resize((cols, rows), Image.LANCZOS)
    z = np.asarray(d, dtype=np.float32) / 255.0

    # Normalisasi tahan outlier (percentile 2–98)
    lo, hi = np.percentile(z, 2.0), np.percentile(z, 98.0)
    z = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    # Gamma untuk angkat mid-tone
    z = np.power(z, 0.85)
    # Redam noise tapi jaga tepi
    z = ndi.median_filter(z, size=3)
    z = ndi.gaussian_filter(z, sigma=0.4)

    aspect = w0 / h0
    width, height = (2.0, 2.0 / aspect) if aspect >= 1 else (2.0 * aspect, 2.0)

    xs = np.linspace(-width / 2, width / 2, cols)
    ys = np.linspace(height / 2, -height / 2, rows)
    X, Y = np.meshgrid(xs, ys)
    Z = z * relief * min(width, height) * 0.9

    vertices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    U, V = np.meshgrid(np.linspace(0, 1, cols), np.linspace(1, 0, rows))
    uv = np.stack([U, V], axis=-1).reshape(-1, 2)

    idx = np.arange(rows * cols).reshape(rows, cols)
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([
        np.stack([a, c, b], -1).reshape(-1, 3),
        np.stack([b, c, e], -1).reshape(-1, 3),
    ])

    depth_back = -0.12 * max(width, height)
    ring = np.concatenate([
        idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0],
    ])
    n = len(ring)
    wall_v = vertices[ring].copy()
    wall_v[:, 2] = depth_back
    wall_uv = uv[ring]

    base = len(vertices)
    k = np.arange(n)
    fi, fj = ring[k], ring[(k + 1) % n]
    bi, bj = base + k, base + (k + 1) % n
    wall_f = np.concatenate([
        np.stack([fi, fj, bi], -1),
        np.stack([fj, bj, bi], -1),
    ])

    hx, hy = width / 2, height / 2
    back_v = np.array([
        [-hx,  hy, depth_back],
        [ hx,  hy, depth_back],
        [-hx, -hy, depth_back],
        [ hx, -hy, depth_back],
    ])
    back_uv = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float64)
    bb = base + n
    back_f = np.array([[bb, bb + 1, bb + 2], [bb + 1, bb + 3, bb + 2]])

    vertices = np.concatenate([vertices, wall_v, back_v])
    uv = np.concatenate([uv, wall_uv, back_uv])
    faces = np.concatenate([faces, wall_f, back_f])

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=img,
        metallicFactor=0.05,
        roughnessFactor=0.85,
        doubleSided=False,
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    mesh.fix_normals()

    glb = mesh.export(file_type="glb")
    print(f"[relief] {len(mesh.vertices)} v, {len(mesh.faces)} f, "
          f"{len(glb) / 1024:.0f} KB, {time.time() - t0:.1f}s")
    return glb


# ==================== HELPERS BERSAMA ====================

def foreground_rgba(img: Image.Image, ratio: float = 0.85) -> Image.Image:
    """Hapus latar belakang (jika perlu), crop ke bounding box, resize ke square."""
    if img.mode in ("RGBA", "LA") and np.asarray(img.convert("RGBA"))[..., 3].min() < 250:
        rgba = img.convert("RGBA")
    else:
        from rembg import new_session, remove
        if "rembg" not in state:
            state["rembg"] = new_session()
        rgba = remove(to_rgb(img), session=state["rembg"]).convert("RGBA")

    alpha = np.asarray(rgba)[..., 3]
    ys, xs = np.nonzero(alpha > 10)
    if len(xs) == 0:
        raise HTTPException(422, "Objek tidak terdeteksi pada gambar.")
    crop = rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = int(max(crop.size) / ratio)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas


# ==================== MODE OBJEK 360° — SF3D (backend utama) ====================

def get_sf3d():
    if "sf3d" in state:
        return state["sf3d"]
    try:
        from sf3d.system import SF3D  # noqa
    except ImportError as err:
        raise HTTPException(
            503,
            "SF3D belum terinstall. Jalankan:\n"
            "pip install --no-deps git+https://github.com/Stability-AI/stable-fast-3d.git\n"
            f"Detail: {err}"
        )

    print("[init] Memuat Stable Fast 3D (run pertama unduh ~1.5 GB)...")
    from sf3d.system import SF3D
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(state["device"])
    model.eval()
    state["sf3d"] = model
    print("[init] SF3D siap.")
    return model


def _sf3d_preprocess(img: Image.Image):
    """Siapkan tensor input untuk SF3D sesuai README resmi."""
    try:
        from sf3d.utils import remove_background, resize_foreground
        pil = img.convert("RGBA")
        pil = remove_background(pil)
        pil = resize_foreground(pil, 0.85)
    except Exception as err:
        print(f"[sf3d] Utils resmi gagal ({err}); pakai preprocessing kita...")
        pil = foreground_rgba(img, ratio=0.85)

    arr = np.array(pil).astype(np.float32) / 255.0
    # Sesuai README resmi SF3D: flip channel dari RGBA ke urutan internal SF3D
    arr = np.flip(arr, axis=-1).copy()
    return torch.from_numpy(arr)


def build_object_sf3d(img: Image.Image, quality: int) -> bytes:
    t0 = time.time()
    model = get_sf3d()

    img_tensor = _sf3d_preprocess(img)
    # Map "quality" (224..320) ke tekstur 1024..2048
    tex_res = int(min(max(quality * 5, 1024), 2048))
    print(f"[sf3d] Input 512x512, texture {tex_res}px")

    def _run(device: str):
        model.to(device)
        t = img_tensor.to(device)
        with torch.no_grad():
            out = model.run_image(
                t,
                bake_resolution=tex_res,
                remesh="none",
                vertex_color="none",
            )
        return out[0] if isinstance(out, tuple) else out

    try:
        mesh = _run(state["device"])
    except Exception as err:
        if state["device"] != "mps":
            raise
        print(f"[sf3d] MPS gagal ({err}); ulangi di CPU...")
        mesh = _run("cpu")
        model.to(state["device"])  # kembalikan ke MPS untuk permintaan berikutnya

    if state["device"] == "mps":
        torch.mps.empty_cache()
    gc.collect()

    glb = mesh.export(file_type="glb")
    print(f"[sf3d] ✅ {len(mesh.vertices)} v, {len(mesh.faces)} f, "
          f"{len(glb) / 1024 / 1024:.2f} MB, {time.time() - t0:.1f}s")
    return glb


# ==================== MODE OBJEK 360° — TripoSR (backend ringan) ====================

def install_mcubes_shim():
    from skimage import measure

    def marching_cubes(vol, thresh):
        v, f, _, _ = measure.marching_cubes(vol.detach().cpu().numpy(), float(thresh))
        v = v[:, ::-1]
        return (
            torch.from_numpy(v.copy()).float().to(vol.device),
            torch.from_numpy(np.ascontiguousarray(f).astype(np.int64)).to(vol.device),
        )

    mod = types.ModuleType("torchmcubes")
    mod.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = mod


def get_triposr():
    if "tsr" in state:
        return state["tsr"]
    if not TRIPOSR_DIR.exists():
        raise HTTPException(
            503,
            "Folder TripoSR belum ada. Jalankan: "
            "git clone https://github.com/VAST-AI-Research/TripoSR.git",
        )
    if str(TRIPOSR_DIR) not in sys.path:
        sys.path.insert(0, str(TRIPOSR_DIR))
    install_mcubes_shim()
    try:
        from tsr.system import TSR
    except ImportError as err:
        raise HTTPException(503, f"Dependensi TripoSR belum lengkap: {err}")

    print("[init] Memuat TripoSR (run pertama unduh ~1,7 GB)...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(2048)  # turun dari 4096 → aman 16 GB
    model.to(state["device"])
    state["tsr"] = model
    print("[init] TripoSR siap.")
    return model


def mesh_adjacency(faces, n):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float64)
    deg = np.asarray(a.sum(1)).ravel()
    return sparse.diags(1.0 / np.maximum(deg, 1.0)) @ a


def taubin_smooth(vertices, P, iterations=5, lamb=0.4, mu=-0.45):
    v = np.asarray(vertices, dtype=np.float64).copy()
    for _ in range(iterations):
        v += lamb * (P @ v - v)
        v += mu * (P @ v - v)
    return v


def project_points(pos, f, cx, cy, d):
    dist = d - pos[:, 2]
    return f * pos[:, 0] / dist + cx, -f * pos[:, 1] / dist + cy, dist


def silhouette(u, v, shape):
    h, w = shape
    m = np.zeros((h, w), bool)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    m[vi[ok], ui[ok]] = True
    return ndi.binary_fill_holes(
        ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=2)
    )


def fit_camera(pos, mask, d0=1.9, iters=250):
    ys, xs = np.nonzero(mask)
    zc, hy = pos[:, 2].mean(), np.ptp(pos[:, 1])
    f = (ys.max() - ys.min()) * (d0 - zc) / hy
    cx = (xs.min() + xs.max()) / 2 - f * pos[:, 0].mean() / (d0 - zc)
    cy = (ys.min() + ys.max()) / 2 + f * pos[:, 1].mean() / (d0 - zc)

    def iou(p):
        s = silhouette(*project_points(pos, *p)[:2], mask.shape)
        return (s & mask).sum() / max((s | mask).sum(), 1)

    p = np.array([f, cx, cy, d0])
    best = iou(p)
    step = np.array([f * 0.05, 8.0, 8.0, 0.3])
    rng = np.random.default_rng(0)
    for i in range(iters):
        q = p + step * rng.normal(size=4) * (rng.random(4) < 0.6)
        q[3] = np.clip(q[3], 1.1, 6.0)
        score = iou(q)
        if score > best:
            p, best = q, score
        if i % 60 == 59:
            step *= 0.6
    return p, best


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def apply_photo_projection(pos, P, nrm, base, rgba, max_side=1280):
    img = Image.fromarray(rgba)
    if max(img.size) > max_side:
        k = max_side / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    rgba = np.asarray(img)
    h, w = rgba.shape[:2]
    fgm = rgba[..., 3] > 128
    cam, iou = fit_camera(pos, ndi.binary_fill_holes(fgm))
    print(f"[triposr] Camera fit IoU: {iou:.3f}")
    if iou < 0.85:
        return base

    rgb = rgba[..., :3] / 255.0
    _, (iy, ix) = ndi.distance_transform_edt(~fgm, return_indices=True)
    rgb = rgb[iy, ix]

    u, v, dist = project_points(pos, *cam)
    inside = (u >= 1) & (u < w - 2) & (v >= 1) & (v < h - 2)
    ui = np.clip(np.round(u).astype(int), 0, w - 1)
    vi = np.clip(np.round(v).astype(int), 0, h - 1)
    zbuf = np.full((h, w), np.inf)
    np.minimum.at(zbuf, (vi[inside], ui[inside]), dist[inside])
    zbuf = ndi.minimum_filter(zbuf, size=3)
    visible = inside & (dist <= zbuf[vi, ui] + 0.02)

    to_cam = np.stack([-pos[:, 0], -pos[:, 1], cam[3] - pos[:, 2]], axis=1)
    to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-8
    facing = smoothstep(0.2, 0.6, (nrm * to_cam).sum(1))
    edge = ndi.gaussian_filter(ndi.binary_erosion(fgm, iterations=3).astype(float), 1.5)
    edge_w = np.clip(ndi.map_coordinates(edge, [v, u], order=1, mode="nearest"), 0, 1)
    weight = visible * facing * edge_w
    for _ in range(2):
        weight = P @ weight
    color = np.stack(
        [ndi.map_coordinates(rgb[..., k], [v, u], order=1, mode="nearest") for k in range(3)],
        axis=1,
    )
    return base * (1 - weight[:, None]) + color * weight[:, None]


def calibrate_colors(colors, nz, photo, strength=0.85):
    w = np.clip(nz, 0, None)
    if len(photo) < 100 or w.sum() < 1e-6:
        return colors
    w = (w / w.sum())[:, None]
    mu_m = (colors * w).sum(0)
    sd_m = np.sqrt((((colors - mu_m) ** 2) * w).sum(0))
    mu_p, sd_p = photo.mean(0), photo.std(0)
    scale = np.clip(sd_p / np.maximum(sd_m, 1e-4), 0.8, 1.25)
    adjusted = (colors - mu_m) * scale + mu_p
    return np.clip(colors + strength * (adjusted - colors), 0.0, 1.0)


def build_object_triposr(img: Image.Image, resolution: int) -> bytes:
    t0 = time.time()
    model = get_triposr()
    fg = np.asarray(foreground_rgba(img))
    arr = fg.astype(np.float32) / 255.0
    arr = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * 0.5
    image = Image.fromarray((arr * 255).astype(np.uint8))
    photo = fg[fg[..., 3] > 128][:, :3] / 255.0

    orig_device = state["device"]

    # ===== Inference (dengan fallback lokal ke CPU) =====
    def _infer(device: str):
        model.to(device)
        with torch.no_grad():
            codes_ = model([image], device=device)
            mesh_ = model.extract_mesh(codes_, True, resolution=resolution)[0]
        return codes_, mesh_

    try:
        codes, mesh = _infer(orig_device)
    except Exception as err:
        if orig_device != "mps":
            raise
        print(f"[triposr] MPS gagal ({err}); ulangi di CPU...")
        codes, mesh = _infer("cpu")
        model.to(orig_device)

    print(f"[triposr] Inference OK ({resolution}³): "
          f"{len(mesh.vertices)} v, {len(mesh.faces)} f")

    # Orientasi standar TripoSR
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))

    # Smoothing ringan (5, bukan 10)
    try:
        P = mesh_adjacency(np.asarray(mesh.faces), len(mesh.vertices))
        mesh.vertices = taubin_smooth(mesh.vertices, P, iterations=5, lamb=0.4, mu=-0.45)
    except Exception as err:
        print(f"[triposr] Smoothing dilewati: {err}")
        P = None

    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_inversion(mesh, multibody=True)
        trimesh.repair.fill_holes(mesh)
        _ = mesh.vertex_normals
    except Exception as err:
        print(f"[triposr] Repair dilewati: {err}")

    # ===== Texture bake — signature RESMI: bake_texture(mesh, scene_codes, texture_resolution) =====
    bake_success = False
    try:
        from tsr.bake_texture import bake_texture

        if orig_device == "mps":
            print("[triposr] Bake texture di CPU (stabil di Apple Silicon)...")
            model_cpu = model.to("cpu")
            code_for_bake = codes[0].detach().to("cpu")
            tex_res = 1024 if resolution <= 224 else 1280
            bake = bake_texture(mesh, code_for_bake, texture_resolution=tex_res)
            model.to(orig_device)
        else:
            tex_res = 1024 if resolution <= 224 else 1536
            bake = bake_texture(mesh, codes[0], texture_resolution=tex_res)

        vmapping = bake["vmapping"]
        indices = bake["indices"]
        uvs = bake["uvs"]
        colors = np.clip(bake["colors"], 0, 1)

        texture_img = Image.fromarray((colors * 255).astype(np.uint8))
        texture_img = texture_img.transpose(Image.FLIP_TOP_BOTTOM)

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_img,
            metallicFactor=0.0,
            roughnessFactor=0.85,
            doubleSided=True,
        )
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices[vmapping],
            faces=indices,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
            process=False,
        )
        print(f"[triposr] ✅ Texture bake OK ({tex_res}px)")
        bake_success = True
    except Exception as err:
        import traceback
        traceback.print_exc()
        print(f"[triposr] ❌ Bake texture gagal: {err}")

    # Fallback: vertex color + proyeksi foto
    if not bake_success:
        try:
            c = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
            c = calibrate_colors(c, mesh.vertex_normals[:, 2], photo, strength=0.85)
            if P is None:
                P = mesh_adjacency(np.asarray(mesh.faces), len(mesh.vertices))
            try:
                c = apply_photo_projection(
                    np.asarray(mesh.vertices), P,
                    np.asarray(mesh.vertex_normals), c, fg, max_side=1280,
                )
            except Exception as e2:
                print(f"[triposr] Proyeksi foto dilewati: {e2}")
            lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
            rgba = np.concatenate([
                np.round(np.clip(lin, 0, 1) * 255),
                np.full((len(c), 1), 255),
            ], axis=1)
            mesh.visual.vertex_colors = rgba.astype(np.uint8)
        except Exception as final_err:
            print(f"[triposr] Fallback bermasalah: {final_err}")

    if orig_device == "mps":
        torch.mps.empty_cache()
    gc.collect()

    glb = mesh.export(file_type="glb")
    print(f"[triposr] Selesai: {len(glb) / 1024 / 1024:.2f} MB, {time.time() - t0:.1f}s")
    return glb


# ==================== API ====================

@app.post("/api/convert")
def convert(
    file: UploadFile = File(...),
    relief: float = Form(0.40),
    detail: int = Form(384),
    mode: str = Form("relief"),
    backend: str = Form("sf3d"),   # "sf3d" atau "triposr"
    quality: int = Form(256),
):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ukuran file maksimal 15 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "File bukan gambar yang valid.")
    if img.format not in {"JPEG", "PNG"}:
        raise HTTPException(400, "Format harus JPG, JPEG, atau PNG.")

    with gpu_lock:
        if mode == "object":
            if backend == "triposr":
                glb = build_object_triposr(img, min(max(quality, 224), 320))
            else:
                glb = build_object_sf3d(img, min(max(quality, 224), 320))
        else:
            relief = min(max(relief, 0.08), 1.0)
            detail = min(max(detail, 96), 640)
            glb = build_relief(img, relief, detail)

    return Response(
        glb,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": 'attachment; filename="model.glb"'},
    )


app.mount(
    "/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static"
)