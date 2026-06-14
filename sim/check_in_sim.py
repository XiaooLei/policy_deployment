"""Drive the dual-YAM MuJoCo scene in a live viewer window or to an mp4.

Two run modes (--mode):

  replay   replays the 14-dim action stream from a pkl bundle
           (output of scripts/extract_mcap.py) directly into MuJoCo.

  policy   at each control step, packs the current robot state +
           the recorded JPEGs from the bundle into the OpenpiClient
           message format and asks a websocket policy server for
           the next action chunk, then executes it in the sim.

Two output modes:

  default     pop up an interactive MuJoCo viewer
              (requires mjpython on macOS).

  --output FILE.mp4  no window; render offscreen and write a video.
                     Works under plain `python` on any platform.

Action / state layout (14-dim, yam-physical convention):
  [0:6]  = left  arm joints 1-6
  [6]    = left  gripper   (1.0 = open, 0.0 = closed; mapped onto MuJoCo
                            gripper joint range [-2.7, 0.0])
  [7:13] = right arm joints 1-6
  [13]   = right gripper

The MuJoCo scene defaults to
  third_party/robot_models/arm/dual_yam/dual_yam_bimanual.xml
whose actuator order is exactly:
  left_act1..6, left_gripper_act, right_act1..6, right_gripper_act.

Usage:
  # Live viewer (macOS: use mjpython so the GLFW window has the main thread)
  mjpython scripts/sim_dual_yam.py --mode replay \\
      --bundle data/example_bundle.pkl

  # Offscreen video (any platform, normal python)
  python   scripts/sim_dual_yam.py --mode replay \\
      --bundle data/example_bundle.pkl \\
      --output out/replay.mp4 --render-camera front --fps 30

  # Policy mode, same toggle
  mjpython scripts/sim_dual_yam.py --mode policy  \\
      --bundle data/example_bundle.pkl \\
      --host 127.0.0.1 --port 8000 --prompt "fold the cloth"
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
import time
from pathlib import Path

# Make the repo root importable so `from scripts._client import ...` works
# regardless of whether the user runs us from the repo root or via an
# absolute path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts._client import WebsocketPolicyClient


# -- gripper convention ------------------------------------------------------
#   normalized g in [0, 1]      (1 = open, 0 = closed)
#   joint ctrl is a slide (m)   (0.0475 = open, 0 = closed)
#   See dual_yam_bimanual.xml: each gripper is a linear_4310 parallel-jaw
#   with two tip bodies coupled by <equality>; *_gripper_act drives the
#   master joint over ctrlrange [0, 0.0475] (≈ 47.5 mm).
GRIP_OPEN_CTRL  = 0.0475
GRIP_CLOSE_CTRL = 0.0


def grip_norm_to_ctrl(g: float) -> float:
    g = float(np.clip(g, 0.0, 1.0))
    return GRIP_OPEN_CTRL * g + GRIP_CLOSE_CTRL * (1.0 - g)


def grip_qpos_to_norm(q: float) -> float:
    """Inverse: ctrl-domain qpos → normalized 0..1, used when packing state."""
    return float(np.clip((q - GRIP_CLOSE_CTRL) / (GRIP_OPEN_CTRL - GRIP_CLOSE_CTRL), 0.0, 1.0))


def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_model(scene_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    return model, data


# ---------------------------------------------------------------------------
# Joint / actuator index helpers
# ---------------------------------------------------------------------------

ARM_JOINTS = [f"{side}_joint{i}" for side in ("left", "right") for i in range(1, 7)]
ACT_NAMES  = (
    [f"left_act{i}"  for i in range(1, 7)] + ["left_gripper_act"]
    + [f"right_act{i}" for i in range(1, 7)] + ["right_gripper_act"]
)
ALL_JOINTS = ARM_JOINTS + ["left_gripper_joint", "right_gripper_joint"]


def _resolve_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Return:
      qpos_addrs: (14,) qpos addresses in the layout
                  [L_j1..6, L_grip, R_j1..6, R_grip]
      ctrl_idxs : (14,) actuator indices in the same layout
    """
    qpos_order = (
        [f"left_joint{i}"  for i in range(1, 7)] + ["left_gripper_joint"]
        + [f"right_joint{i}" for i in range(1, 7)] + ["right_gripper_joint"]
    )
    qpos_addrs = []
    for n in qpos_order:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise RuntimeError(f"Joint {n} not in scene")
        qpos_addrs.append(model.jnt_qposadr[jid])

    ctrl_idxs = []
    for n in ACT_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if aid < 0:
            raise RuntimeError(f"Actuator {n} not in scene")
        ctrl_idxs.append(aid)

    return np.asarray(qpos_addrs, dtype=np.int32), np.asarray(ctrl_idxs, dtype=np.int32)


# ---------------------------------------------------------------------------
# 14-dim action → ctrl
# ---------------------------------------------------------------------------

def action14_to_ctrl(action14: np.ndarray, ctrl_idxs: np.ndarray, data: mujoco.MjData) -> None:
    a = np.asarray(action14, dtype=np.float64).flatten()
    assert a.shape == (14,), f"expected (14,), got {a.shape}"
    # arms
    data.ctrl[ctrl_idxs[0:6]]  = a[0:6]
    data.ctrl[ctrl_idxs[7:13]] = a[7:13]
    # grippers (normalized → ctrl space)
    data.ctrl[ctrl_idxs[6]]  = grip_norm_to_ctrl(a[6])
    data.ctrl[ctrl_idxs[13]] = grip_norm_to_ctrl(a[13])


def read_state14(qpos_addrs: np.ndarray, data: mujoco.MjData) -> np.ndarray:
    """Inverse of action14_to_ctrl: read 14-dim state in normalized convention."""
    qp = data.qpos[qpos_addrs]
    state = np.empty(14, dtype=np.float64)
    state[0:6]  = qp[0:6]
    state[6]    = grip_qpos_to_norm(qp[6])
    state[7:13] = qp[7:13]
    state[13]   = grip_qpos_to_norm(qp[13])
    return state


# ---------------------------------------------------------------------------
# Output sinks: live viewer  OR  offscreen video writer.
# Both expose .is_running() and .sync() so run_replay/run_policy don't care.
# ---------------------------------------------------------------------------

class _VideoSink:
    """Offscreen renderer + mp4 writer with the same minimal API as
    mujoco.viewer.Handle: .is_running() and .sync()."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *,
                 out_path: Path, camera: str,
                 width: int, height: int, fps: int):
        import imageio  # local import so video deps stay optional
        self.model = model
        self.data = data
        self.camera = camera
        self.fps = fps
        self.dt_video = 1.0 / fps
        self.next_t = 0.0
        self.frames = 0
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(out_path), fps=fps, codec="libx264", quality=8,
            macro_block_size=1,
        )
        self._closed = False
        self._stopped = False
        self.out_path = out_path

    def is_running(self) -> bool:
        return not self._stopped

    def stop(self) -> None:
        self._stopped = True

    def sync(self) -> None:
        """Capture a frame iff enough sim time has elapsed for the target fps."""
        if self.data.time + 1e-9 < self.next_t:
            return
        self.renderer.update_scene(self.data, camera=self.camera)
        frame = self.renderer.render()
        self.writer.append_data(frame)
        self.frames += 1
        self.next_t += self.dt_video

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.writer.close()
        self.renderer.close()
        print(f"Wrote {self.out_path}  frames={self.frames}  fps={self.fps}")


def init_to_first_frame(model, data, qpos_addrs, ctrl_idxs, bundle,
                        source: str = "actions") -> None:
    """Snap qpos AND ctrl to the first frame of `source` so the viewer
    starts in the commanded pose (no P-controller jolt on step 1).

    source can be 'actions' (default; matches what we'll start streaming)
    or 'joint_positions' (proprio at t=0).
    """
    if source == "joint_positions":
        v0 = bundle["joint_positions"][0]
    else:
        v0 = bundle["actions"][0]
    data.qpos[qpos_addrs[0:6]]  = v0[0:6]
    data.qpos[qpos_addrs[7:13]] = v0[7:13]
    data.qpos[qpos_addrs[6]]    = grip_norm_to_ctrl(v0[6])
    data.qpos[qpos_addrs[13]]   = grip_norm_to_ctrl(v0[13])
    # Match actuator targets so the position controller doesn't pull qpos
    # toward 0 on the very first step.
    data.ctrl[ctrl_idxs[0:6]]   = v0[0:6]
    data.ctrl[ctrl_idxs[7:13]]  = v0[7:13]
    data.ctrl[ctrl_idxs[6]]     = grip_norm_to_ctrl(v0[6])
    data.ctrl[ctrl_idxs[13]]    = grip_norm_to_ctrl(v0[13])
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Mode: replay
# ---------------------------------------------------------------------------

def run_replay(model, data, viewer, qpos_addrs, ctrl_idxs, bundle, *,
               speed: float = 1.0, source: str = "actions",
               ctrl_hz: float = 60.0, realtime: bool = True) -> None:
    actions = bundle[source]  # (T, 14)
    T       = len(actions)
    sim_dt = model.opt.timestep
    ctrl_dt = 1.0 / ctrl_hz
    inner_steps = max(1, int(round(ctrl_dt / sim_dt)))
    print(f"Replay: {T} frames from {source}  "
          f"ctrl_hz={ctrl_hz:.1f}  sim_dt={sim_dt:.4f}s  "
          f"inner_steps={inner_steps}  realtime={realtime}")

    t_wall0 = time.perf_counter()
    for k in range(T):
        if not viewer.is_running():
            break
        if realtime:
            target_wall = t_wall0 + (k * ctrl_dt) / max(speed, 1e-6)
            sleep_for = target_wall - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
        action14_to_ctrl(actions[k], ctrl_idxs, data)
        for _ in range(inner_steps):
            mujoco.mj_step(model, data)
        viewer.sync()
    print("Replay done.")


# ---------------------------------------------------------------------------
# Mode: policy
# ---------------------------------------------------------------------------

def jpeg_to_chw(jpeg_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)            # (H, W, 3)
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))  # (3, H, W)


def policy_images_from_bundle(images: dict, *, send_compressed_images: bool = False) -> dict:
    """Map bundle camera names to policy camera names.

    By default this preserves the original behavior and sends decoded CHW uint8
    arrays. With send_compressed_images=True it sends the JPEG bytes stored in
    the slim bundle directly; patched OpenPI servers decode them on receipt,
    reducing WAN payload size by roughly 10x.
    """

    if send_compressed_images:
        return {
            "cam_high": images["top_camera"],
            "cam_left_wrist": images["left_camera"],
            "cam_right_wrist": images["right_camera"],
        }
    return {
        "cam_high": jpeg_to_chw(images["top_camera"]),
        "cam_left_wrist": jpeg_to_chw(images["left_camera"]),
        "cam_right_wrist": jpeg_to_chw(images["right_camera"]),
    }


def run_policy(model, data, viewer, qpos_addrs, ctrl_idxs, bundle, *,
               host: str, port: int, prompt: str,
               api_key: str | None = None,
               action_horizon: int = 10, speed: float = 1.0,
               ctrl_hz: float = 60.0, realtime: bool = True,
               send_compressed_images: bool = False) -> None:
    client = WebsocketPolicyClient(host=host, port=port, api_key=api_key)
    print(f"Connected to policy server at {host}:{port}")

    images_top   = bundle["images"]["top_camera"]
    images_left  = bundle["images"]["left_camera"]
    images_right = bundle["images"]["right_camera"]
    T = len(bundle["timestamps_ns"])

    sim_dt = model.opt.timestep
    ctrl_dt = 1.0 / ctrl_hz
    inner_steps = max(1, int(round(ctrl_dt / sim_dt)))
    print(f"Policy: T={T}  ctrl_hz={ctrl_hz:.1f}  "
          f"sim_dt={sim_dt:.4f}s  inner_steps={inner_steps}  realtime={realtime}")

    t_wall0 = time.perf_counter()
    k = 0
    while viewer.is_running() and k < T:
        # ---- obs at frame k (mcap JPEG for vision, sim qpos for state) ----
        state14 = read_state14(qpos_addrs, data).astype(np.float64)
        policy_input = {
            "images": policy_images_from_bundle(
                {
                    "top_camera": images_top[k],
                    "left_camera": images_left[k],
                    "right_camera": images_right[k],
                },
                send_compressed_images=send_compressed_images,
            ),
            "state":  np.ascontiguousarray(state14),
            "prompt": prompt,
        }
        chunk = np.asarray(client.infer(policy_input)["actions"])  # (H, 14)
        H = min(action_horizon, len(chunk))
        print(f"frame {k:4d}/{T}  policy returned {len(chunk)} actions, executing {H}")

        for h in range(H):
            if not viewer.is_running() or k >= T:
                break
            if realtime:
                target_wall = t_wall0 + (k * ctrl_dt) / max(speed, 1e-6)
                sleep_for = target_wall - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
            action14_to_ctrl(chunk[h], ctrl_idxs, data)
            for _ in range(inner_steps):
                mujoco.mj_step(model, data)
            viewer.sync()
            k += 1
    print("Policy run done.")


# ---------------------------------------------------------------------------
# Mode: compare  (slim bundle only) — for each sample, run the recorded
# action chunk and the policy-predicted chunk back-to-back so you can
# eyeball the difference in the viewer / video.
# ---------------------------------------------------------------------------

def _snap_to_state14(model, data, qpos_addrs, ctrl_idxs,
                     state14: np.ndarray) -> None:
    v0 = np.asarray(state14, dtype=np.float64)
    data.qpos[qpos_addrs[0:6]]  = v0[0:6]
    data.qpos[qpos_addrs[7:13]] = v0[7:13]
    data.qpos[qpos_addrs[6]]    = grip_norm_to_ctrl(v0[6])
    data.qpos[qpos_addrs[13]]   = grip_norm_to_ctrl(v0[13])
    data.ctrl[ctrl_idxs[0:6]]   = v0[0:6]
    data.ctrl[ctrl_idxs[7:13]]  = v0[7:13]
    data.ctrl[ctrl_idxs[6]]     = grip_norm_to_ctrl(v0[6])
    data.ctrl[ctrl_idxs[13]]    = grip_norm_to_ctrl(v0[13])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _run_chunk(model, data, viewer, ctrl_idxs, chunk: np.ndarray, *,
               ctrl_hz: float, realtime: bool, speed: float,
               label: str = "") -> None:
    """Execute one action chunk (H, 14) in sim, syncing the sink each step."""
    sim_dt = model.opt.timestep
    ctrl_dt = 1.0 / ctrl_hz
    inner_steps = max(1, int(round(ctrl_dt / sim_dt)))
    if label:
        viewer.sync()  # render at least one frame of "pre" state with label
    t_wall0 = time.perf_counter()
    for h in range(len(chunk)):
        if not viewer.is_running():
            return
        if realtime:
            target_wall = t_wall0 + (h * ctrl_dt) / max(speed, 1e-6)
            sleep_for = target_wall - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
        action14_to_ctrl(chunk[h], ctrl_idxs, data)
        for _ in range(inner_steps):
            mujoco.mj_step(model, data)
        viewer.sync()


def run_compare_slim(model, data, viewer, qpos_addrs, ctrl_idxs, bundle, *,
                     host: str, port: int, prompt: str,
                     api_key: str | None = None,
                     action_horizon: int | None = None,
                     speed: float = 1.0,
                     ctrl_hz: float = 60.0, realtime: bool = True,
                     pause_s: float = 0.5,
                     send_compressed_images: bool = False) -> None:
    """For each sample in a slim bundle (output of make_slim_bundle.py):

      1. Snap sim to recorded state, replay the recorded action chunk.
      2. Snap sim back, query the policy, execute the predicted chunk.
      3. Print L2 / max-abs diff between predicted and recorded chunks.
    """
    client = WebsocketPolicyClient(host=host, port=port, api_key=api_key)
    print(f"Connected to policy server at {host}:{port}")

    samples = bundle["samples"]
    print(f"Compare (slim): {len(samples)} samples  "
          f"ctrl_hz={ctrl_hz:.1f}  realtime={realtime}  pause_s={pause_s}")

    for s_idx, s in enumerate(samples):
        if not viewer.is_running():
            break
        recorded = np.asarray(s["action_chunk"])
        H_rec = len(recorded) if action_horizon is None else min(action_horizon, len(recorded))
        recorded = recorded[:H_rec]

        # ---- 1. Recorded chunk in sim ----
        print(f"\nsample[{s_idx:2d}] frame={s['frame_idx']:4d}  "
              f"-- recorded chunk ({H_rec} actions)")
        _snap_to_state14(model, data, qpos_addrs, ctrl_idxs, s["state"])
        if realtime and pause_s > 0:
            t0 = time.perf_counter()
            while viewer.is_running() and time.perf_counter() - t0 < pause_s:
                viewer.sync(); time.sleep(0.02)
        _run_chunk(model, data, viewer, ctrl_idxs, recorded,
                   ctrl_hz=ctrl_hz, realtime=realtime, speed=speed,
                   label=f"sample{s_idx}-recorded")

        # ---- 2. Policy rollout from same starting state ----
        _snap_to_state14(model, data, qpos_addrs, ctrl_idxs, s["state"])
        if realtime and pause_s > 0:
            t0 = time.perf_counter()
            while viewer.is_running() and time.perf_counter() - t0 < pause_s:
                viewer.sync(); time.sleep(0.02)

        state_for_policy = np.ascontiguousarray(
            np.asarray(s["state"], dtype=np.float64))
        policy_input = {
            "images": policy_images_from_bundle(s["images"], send_compressed_images=send_compressed_images),
            "state":  state_for_policy,
            "prompt": prompt,
        }
        chunk = np.asarray(client.infer(policy_input)["actions"])  # (H, 14)
        H_pol = len(chunk) if action_horizon is None else min(action_horizon, len(chunk))
        chunk = chunk[:H_pol]
        print(f"sample[{s_idx:2d}] frame={s['frame_idx']:4d}  "
              f"-- policy chunk   ({H_pol} actions)")
        _run_chunk(model, data, viewer, ctrl_idxs, chunk,
                   ctrl_hz=ctrl_hz, realtime=realtime, speed=speed,
                   label=f"sample{s_idx}-policy")

        # ---- 3. Diff metrics over the overlap ----
        H_cmp = min(len(recorded), len(chunk))
        if H_cmp > 0:
            diff = (chunk[:H_cmp] - recorded[:H_cmp])
            l2 = float(np.linalg.norm(diff))
            l2_per = float(np.sqrt(np.mean(diff * diff)))
            mx = float(np.max(np.abs(diff)))
            mx_per_dim = np.max(np.abs(diff), axis=0)
            print(f"sample[{s_idx:2d}]  diff over {H_cmp} steps:  "
                  f"L2={l2:.4f}  L2/sqrt(N*14)={l2_per:.4f}  "
                  f"max|.|={mx:.4f}")
            print(f"            max|.| per dim: "
                  + ' '.join(f'{v:.3f}' for v in mx_per_dim))
    print("\nCompare run done.")


# ---------------------------------------------------------------------------
# Mode: compare → side-by-side mp4
# Two independent sims, REPLAY on the left, POLICY on the right, with the
# current sample/frame labelled at the top. Shorter side freezes on its last
# rendered frame until the longer side finishes.
# ---------------------------------------------------------------------------

def _load_overlay_font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int],
               font: ImageFont.ImageFont) -> None:
    x, y = xy
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))


def _text_width(draw: ImageDraw.ImageDraw, text: str,
                font: ImageFont.ImageFont) -> int:
    try:
        l, _, r, _ = draw.textbbox((0, 0), text, font=font)
        return r - l
    except AttributeError:  # very old Pillow
        return draw.textsize(text, font=font)[0]


def _format_action_lines(action: np.ndarray) -> list[str]:
    a = np.asarray(action, dtype=np.float64).flatten()
    return [
        f"L_arm:  " + " ".join(f"{a[i]:+.3f}" for i in range(6)),
        f"L_grip: {a[6]:+.3f}",
        f"R_arm:  " + " ".join(f"{a[i]:+.3f}" for i in range(7, 13)),
        f"R_grip: {a[13]:+.3f}",
    ]


class _DualVideoSink:
    """Side-by-side dual-sim mp4 writer for compare mode."""

    def __init__(self, model_L: mujoco.MjModel, data_L: mujoco.MjData,
                 model_R: mujoco.MjModel, data_R: mujoco.MjData, *,
                 out_path: Path, camera: str,
                 panel_width: int, panel_height: int, fps: int,
                 bar_height: int = 100):
        import imageio
        self.data_L, self.data_R = data_L, data_R
        self.camera = camera
        self.fps = fps
        self.dt_video = 1.0 / fps
        self.next_t = 0.0
        self.frames = 0
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.bar_height = bar_height
        self.renderer_L = mujoco.Renderer(model_L, height=panel_height, width=panel_width)
        self.renderer_R = mujoco.Renderer(model_R, height=panel_height, width=panel_width)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(out_path), fps=fps, codec="libx264", quality=8,
            macro_block_size=1,
        )
        self.label_top = ""  # e.g. "sample 3  frame 152"
        self.font_big = _load_overlay_font(max(16, int(panel_height * 0.07)))
        self.font_small = _load_overlay_font(max(14, int(panel_height * 0.05)))
        _bar_pt = max(10, int(bar_height * 0.11))
        _bar_info_pt = max(11, int(bar_height * 0.13))
        self.font_bar = _load_overlay_font(_bar_pt)
        self.font_bar_info = _load_overlay_font(_bar_info_pt)
        self._bar_line_h = int(_bar_pt * 1.45)
        self._bar_info_h = _bar_info_pt
        self._closed = False
        self.out_path = out_path

    def render_L(self) -> np.ndarray:
        self.renderer_L.update_scene(self.data_L, camera=self.camera)
        return self.renderer_L.render()

    def render_R(self) -> np.ndarray:
        self.renderer_R.update_scene(self.data_R, camera=self.camera)
        return self.renderer_R.render()

    def _compose(self, frame_L: np.ndarray, frame_R: np.ndarray,
                 action_L: np.ndarray | None = None,
                 action_R: np.ndarray | None = None,
                 t: int | None = None) -> np.ndarray:
        # Top: side-by-side renders with labels.
        composite = np.concatenate([frame_L, frame_R], axis=1)
        img = Image.fromarray(composite)
        draw = ImageDraw.Draw(img)
        _draw_text(draw, "REPLAY", (12, 10), self.font_small)
        _draw_text(draw, "POLICY", (self.panel_width + 12, 10), self.font_small)
        if self.label_top:
            tw = _text_width(draw, self.label_top, self.font_big)
            x = self.panel_width - tw // 2  # composite center
            _draw_text(draw, self.label_top, (x, 10), self.font_big)

        # Bottom: action value bar.
        bar_w = self.panel_width * 2
        bar = np.full((self.bar_height, bar_w, 3), 30, dtype=np.uint8)
        bar_img = Image.fromarray(bar)
        bar_draw = ImageDraw.Draw(bar_img)

        # Divider line at the center.
        xx = self.panel_width
        for y in range(self.bar_height):
            bar_img.putpixel((xx, y), (80, 80, 80))

        bar_y_top = 6

        # Left side — REPLAY actions.
        if action_L is not None:
            lines = _format_action_lines(action_L)
            for i, line in enumerate(lines):
                bar_draw.text((10, bar_y_top + i * self._bar_line_h), line,
                              font=self.font_bar, fill=(220, 220, 220))

        # Right side — POLICY actions.
        if action_R is not None:
            lines = _format_action_lines(action_R)
            for i, line in enumerate(lines):
                bar_draw.text((self.panel_width + 10, bar_y_top + i * self._bar_line_h),
                              line, font=self.font_bar, fill=(220, 220, 220))

        # Info line at bottom of bar (t=...  sample... frame...).
        info = f"t={t}" if t is not None else ""
        if info and self.label_top:
            info_full = f"{info}  {self.label_top}"
            y_info = self.bar_height - self._bar_info_h - 5
            # Left half info.
            bar_draw.text((10, y_info), info_full,
                          font=self.font_bar_info, fill=(180, 180, 180))
            # Right half info.
            bar_draw.text((self.panel_width + 10, y_info), info_full,
                          font=self.font_bar_info, fill=(180, 180, 180))

        # Stack: top renders + bottom bar.
        result = np.concatenate([np.asarray(img), np.asarray(bar_img)], axis=0)
        return np.asarray(result)

    def write_pair(self, frame_L: np.ndarray, frame_R: np.ndarray,
                   action_L: np.ndarray | None = None,
                   action_R: np.ndarray | None = None,
                   t: int | None = None) -> None:
        self.writer.append_data(
            self._compose(frame_L, frame_R, action_L, action_R, t))
        self.frames += 1

    def write_static(self, frame_L: np.ndarray, frame_R: np.ndarray,
                     duration_s: float,
                     action_L: np.ndarray | None = None,
                     action_R: np.ndarray | None = None,
                     t: int | None = None) -> None:
        n = max(1, int(round(duration_s * self.fps)))
        for _ in range(n):
            self.write_pair(frame_L, frame_R, action_L, action_R, t)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.writer.close()
        self.renderer_L.close()
        self.renderer_R.close()
        print(f"Wrote {self.out_path}  frames={self.frames}  fps={self.fps}")


def run_compare_slim_dual_video(scene_path: Path, bundle: dict, *,
                                 host: str, port: int, prompt: str,
                                 api_key: str | None = None,
                                 action_horizon: int | None = None,
                                 out_path: Path,
                                 camera: str,
                                 panel_width: int, panel_height: int,
                                 fps: int,
                                 ctrl_hz: float = 60.0,
                                 transition_s: float = 0.8,
                                 send_compressed_images: bool = False) -> None:
    client = WebsocketPolicyClient(host=host, port=port, api_key=api_key)
    print(f"Connected to policy server at {host}:{port}")

    model_L, data_L = build_model(scene_path)
    model_R, data_R = build_model(scene_path)
    qpos_L, ctrl_L = _resolve_indices(model_L)
    qpos_R, ctrl_R = _resolve_indices(model_R)

    samples = bundle["samples"]
    sim_dt = model_L.opt.timestep
    ctrl_dt = 1.0 / ctrl_hz
    inner_steps = max(1, int(round(ctrl_dt / sim_dt)))
    print(f"Compare (dual video): {len(samples)} samples  "
          f"ctrl_hz={ctrl_hz:.1f}  sim_dt={sim_dt:.4f}s  "
          f"inner_steps={inner_steps}  fps={fps}  "
          f"panel={panel_width}x{panel_height}")

    sink = _DualVideoSink(model_L, data_L, model_R, data_R,
                          out_path=out_path, camera=camera,
                          panel_width=panel_width, panel_height=panel_height,
                          fps=fps)

    with sink:
        for s_idx, s in enumerate(samples):
            recorded = np.asarray(s["action_chunk"])
            H_rec = len(recorded) if action_horizon is None else min(action_horizon, len(recorded))
            recorded = recorded[:H_rec]

            # Reset both sims to the recorded state; reset sim clocks so
            # each sample's video-rate gating starts at t=0.
            _snap_to_state14(model_L, data_L, qpos_L, ctrl_L, s["state"])
            _snap_to_state14(model_R, data_R, qpos_R, ctrl_R, s["state"])
            data_L.time = 0.0
            data_R.time = 0.0
            sink.next_t = 0.0
            sink.label_top = f"sample {s_idx}  frame {s['frame_idx']}"
            print(f"\nsample[{s_idx:2d}] frame={s['frame_idx']:4d}")

            # Title / transition card: hold the pre-roll pose with the label.
            pre_L = sink.render_L()
            pre_R = sink.render_R()
            sink.write_static(pre_L, pre_R, transition_s,
                              action_L=recorded[0], action_R=None, t=0)

            # Query policy from the same starting state.
            state_for_policy = np.ascontiguousarray(
                np.asarray(s["state"], dtype=np.float64))
            policy_input = {
                "images": policy_images_from_bundle(s["images"], send_compressed_images=send_compressed_images),
                "state":  state_for_policy,
                "prompt": prompt,
            }
            chunk = np.asarray(client.infer(policy_input)["actions"])
            H_pol = len(chunk) if action_horizon is None else min(action_horizon, len(chunk))
            chunk = chunk[:H_pol]
            print(f"  recorded={H_rec} actions  policy={H_pol} actions")

            H_cmp = min(H_rec, H_pol)
            if H_cmp > 0:
                diff = chunk[:H_cmp] - recorded[:H_cmp]
                l2 = float(np.linalg.norm(diff))
                mx = float(np.max(np.abs(diff)))
                print(f"  diff over {H_cmp} steps: L2={l2:.4f}  max|.|={mx:.4f}")

            # Per-tick dual rollout. Each side only steps while it still has
            # actions; once exhausted, its last rendered frame is reused so
            # the shorter side freezes on screen.
            H_max = max(H_rec, H_pol)
            last_L = pre_L
            last_R = pre_R
            for t in range(H_max):
                if t < H_rec:
                    action14_to_ctrl(recorded[t], ctrl_L, data_L)
                    for _ in range(inner_steps):
                        mujoco.mj_step(model_L, data_L)
                if t < H_pol:
                    action14_to_ctrl(chunk[t], ctrl_R, data_R)
                    for _ in range(inner_steps):
                        mujoco.mj_step(model_R, data_R)
                # Gate frame capture by sim time so fps stays independent of ctrl_hz.
                if data_L.time + 1e-9 < sink.next_t:
                    continue
                if t < H_rec:
                    last_L = sink.render_L()
                if t < H_pol:
                    last_R = sink.render_R()
                act_L = recorded[t] if t < H_rec else None
                act_R = chunk[t] if t < H_pol else None
                sink.write_pair(last_L, last_R, action_L=act_L, action_R=act_R, t=t)
                sink.next_t += sink.dt_video

    print("\nCompare (dual video) run done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("replay", "policy", "compare"), required=True,
                   help="replay: stream actions in sim. policy: query server "
                        "frame-by-frame. compare (slim bundle only): per-sample "
                        "recorded chunk + policy rollout side-by-side.")
    p.add_argument("--bundle", type=Path, default=Path("sim/assets/example_slim.pkl"),
                   help="pkl produced by scripts/extract_mcap.py")
    p.add_argument("--scene",  type=Path,
                   default=Path("sim/assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml"))
    p.add_argument("--speed",  type=float, default=1.0,
                   help="Playback speed multiplier (1.0 = realtime)")
    # replay-only
    p.add_argument("--source", choices=("actions", "gello_actions"), default="actions",
                   help="Which 14-dim stream to replay (default robot actions)")
    # policy-only
    p.add_argument("--host",    type=str, default="127.0.0.1")
    p.add_argument("--port",    type=int, default=8000)
    p.add_argument("--api-key", type=str, default=None,
                   help="API key for the policy server (omit for unauthenticated local runs)")
    p.add_argument("--prompt",  type=str, default="")
    p.add_argument("--action-horizon", type=int, default=50,
                   help="How many actions from each policy chunk to execute "
                        "before re-querying (default 10)")
    # output toggle: live viewer vs. offscreen video
    p.add_argument("--output", type=Path, default=None,
                   help="If set, render offscreen to this mp4 instead of opening the "
                        "live MuJoCo viewer.")
    p.add_argument("--render-camera", type=str, default="front",
                   help="Named scene camera to render in video mode "
                        "(scene provides: front / top / side_left)")
    p.add_argument("--render-width",  type=int, default=640)
    p.add_argument("--render-height", type=int, default=480)
    p.add_argument("--fps",           type=int, default=30,
                   help="Video framerate (video mode only)")
    p.add_argument("--ctrl-hz",       type=float, default=60.0,
                   help="Control loop frequency in Hz (default 60)")
    p.add_argument("--send-compressed-images", action="store_true",
                   help="Send JPEG bytes from the bundle instead of decoded CHW uint8 arrays. "
                        "Requires a server that advertises accepts_compressed_images.")
    # compare-mode specific
    p.add_argument("--pause-s",       type=float, default=0.5,
                   help="In compare mode (viewer only), seconds to pause between "
                        "recorded and policy phases of each sample.")
    args = p.parse_args()

    print(f"Loading bundle {args.bundle}")
    bundle = load_bundle(args.bundle)
    print(f"  T={bundle['meta']['n_frames']}  duration={bundle['meta']['duration_s']:.1f}s")

    # Compare + video output takes the dedicated dual-sim, side-by-side path.
    if args.mode == "compare" and args.output is not None:
        if "samples" not in bundle:
            raise SystemExit(
                "--mode compare requires a slim bundle "
                "(produced by scripts/make_slim_bundle.py)")
        print(f"Compare video mode: writing side-by-side {args.output}  "
              f"camera={args.render_camera} "
              f"panel={args.render_width}x{args.render_height} @ {args.fps}fps")
        run_compare_slim_dual_video(
            args.scene, bundle,
            host=args.host, port=args.port, prompt=args.prompt,
            api_key=args.api_key,
            action_horizon=None if args.action_horizon <= 0 else args.action_horizon,
            out_path=args.output,
            camera=args.render_camera,
            panel_width=args.render_width,
            panel_height=args.render_height,
            fps=args.fps,
            ctrl_hz=args.ctrl_hz,
            transition_s=max(args.pause_s, 0.0) + 0.3,
            send_compressed_images=args.send_compressed_images,
        )
        return

    print(f"Loading scene {args.scene}")
    model, data = build_model(args.scene)
    qpos_addrs, ctrl_idxs = _resolve_indices(model)
    init_to_first_frame(model, data, qpos_addrs, ctrl_idxs, bundle, source="actions")

    if args.output is not None:
        print(f"Video mode: writing {args.output}  "
              f"camera={args.render_camera} {args.render_width}x{args.render_height} @ {args.fps}fps")
        sink_ctx = _VideoSink(model, data, out_path=args.output,
                              camera=args.render_camera,
                              width=args.render_width, height=args.render_height,
                              fps=args.fps)
    else:
        print("Launching MuJoCo viewer...")
        sink_ctx = mujoco.viewer.launch_passive(model, data)

    # Video mode runs deterministically off sim time; viewer mode throttles
    # to wallclock so the user actually sees realtime motion.
    realtime = args.output is None

    with sink_ctx as viewer:
        if args.mode == "replay":
            run_replay(model, data, viewer, qpos_addrs, ctrl_idxs, bundle,
                       speed=args.speed, source=args.source,
                       ctrl_hz=args.ctrl_hz, realtime=realtime)
        elif args.mode == "policy":
            run_policy(model, data, viewer, qpos_addrs, ctrl_idxs, bundle,
                       host=args.host, port=args.port, prompt=args.prompt,
                       api_key=args.api_key,
                       action_horizon=args.action_horizon, speed=args.speed,
                       ctrl_hz=args.ctrl_hz, realtime=realtime,
                       send_compressed_images=args.send_compressed_images)
        elif args.mode == "compare":
            if "samples" not in bundle:
                raise SystemExit(
                    "--mode compare requires a slim bundle "
                    "(produced by scripts/make_slim_bundle.py)")
            run_compare_slim(model, data, viewer, qpos_addrs, ctrl_idxs, bundle,
                             host=args.host, port=args.port, prompt=args.prompt,
                             api_key=args.api_key,
                             action_horizon=None if args.action_horizon <= 0
                                            else args.action_horizon,
                             speed=args.speed, ctrl_hz=args.ctrl_hz,
                             realtime=realtime, pause_s=args.pause_s,
                             send_compressed_images=args.send_compressed_images)


if __name__ == "__main__":
    main()
