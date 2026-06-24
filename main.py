"""Gesture Photo Puzzle Booth — минималистичная версия.

Видео на весь экран, лёгкие полупрозрачные панели (без тяжёлого блюра).
Цикл: FRAME (две руки обводят область) -> двойной щипок -> COUNTDOWN 3с ->
снимок -> REVIEW -> PUZZLE (щипком меняешь куски) -> SOLVED -> ... ->
после NUM_SHOTS: DONE (вертикальная лента на Рабочий стол).
Выход 'q'. В DONE: 'r' — заново.

Запуск из Terminal:
    cd "<папка проекта>" && ./.venv/bin/python main.py
"""
import os
import random
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
CAMERA_INDEX = 1
CAM_WIDTH = 1280
CAM_HEIGHT = 720
WINDOW_NAME = "Photo Puzzle Booth"
OUTPUT_DIR = "output"
DESKTOP_DIR = os.path.expanduser("~/Desktop")

MAX_NUM_HANDS = 2
MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
PROCESS_WIDTH = 600

# Жесты
PINCH_ON = 0.32
PINCH_OFF = 0.50
RATIO_EMA = 0.5          # сглаживание значения щипка
CURSOR_EMA = 0.6         # курсор: больше = отзывчивее (меньше лага)

FRAME_PAD = 0.04
MIN_FRAME_FRAC = 0.12
RECT_EMA = 0.4
COUNTDOWN_SEC = 3
REVIEW_SEC = 1.2
FLASH_SEC = 0.16

PUZZLE_N = 3
BOARD_FILL = 0.66
CELL_GAP = 6
SOLVED_SEC = 1.8

NUM_SHOTS = 3
STRIP_W = 700
STRIP_PAD = 30
STRIP_GAP = 22

# Палитра Apple (BGR)
COL_TEXT = (247, 245, 245)
COL_DIM = (205, 196, 190)
COL_ACCENT = (255, 138, 20)     # systemBlue
COL_SUCCESS = (90, 210, 50)     # systemGreen
COL_WARN = (10, 214, 255)       # systemYellow
TINT = (22, 18, 16)             # тон полупрозрачных панелей
STRIP_BG = (250, 248, 245)

FONT_SANS = "/System/Library/Fonts/SFNS.ttf"
FONT_ROUND = "/System/Library/Fonts/SFNSRounded.ttf"
if not os.path.exists(FONT_SANS):
    FONT_SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
if not os.path.exists(FONT_ROUND):
    FONT_ROUND = FONT_SANS
_font_cache = {}
_mask_cache = {}

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
FRAME_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
LM_SPEC = mp_draw.DrawingSpec(color=(255, 190, 110), thickness=1, circle_radius=2)
CONN_SPEC = mp_draw.DrawingSpec(color=(210, 210, 210), thickness=1)


# ---------------------------------------------------------------------------
# Текст (SF Pro)
# ---------------------------------------------------------------------------
def get_font(px, weight="Regular", rounded=False):
    key = (int(px), weight, rounded)
    if key not in _font_cache:
        f = ImageFont.truetype(FONT_ROUND if rounded else FONT_SANS,
                               max(8, int(px)))
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        _font_cache[key] = f
    return _font_cache[key]


def T(texts, text, x, y, px, color, anchor="tl", dy=0, weight="Regular",
      rounded=False):
    texts.append((text, x, y, px, color, anchor, dy, weight, rounded))


def text_w(text, px, weight="Regular", rounded=False):
    bb = get_font(px, weight, rounded).getbbox(text)
    return bb[2] - bb[0]


def render_texts(canvas, items, cam_w, cam_h):
    if not items:
        return canvas
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for text, x, y, px, color, anchor, dy, weight, rounded in items:
        font = get_font(px, weight, rounded)
        fill = (int(color[2]), int(color[1]), int(color[0]))
        bbox = d.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if anchor == "center":
            x, y = (cam_w - tw) // 2, (cam_h - th) // 2 + dy
        elif anchor == "cx":
            x = (cam_w - tw) // 2
        d.text((x + 1, y + 1), text, font=font, fill=(12, 12, 12))
        d.text((x, y), text, font=font, fill=fill)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Лёгкие панели (без блюра)
# ---------------------------------------------------------------------------
def _clamp(img, x0, y0, x1, y1):
    return (max(0, int(x0)), max(0, int(y0)),
            min(img.shape[1], int(x1)), min(img.shape[0], int(y1)))


def rounded_mask(h, w, r):
    key = (h, w, int(r))
    m = _mask_cache.get(key)
    if m is None:
        a = np.zeros((h, w), np.uint8)
        rr = max(0, min(int(r), w // 2, h // 2))
        cv2.rectangle(a, (rr, 0), (w - rr, h), 255, -1)
        cv2.rectangle(a, (0, rr), (w, h - rr), 255, -1)
        for cx, cy in ((rr, rr), (w - rr, rr), (rr, h - rr), (w - rr, h - rr)):
            cv2.circle(a, (cx, cy), rr, 255, -1, cv2.LINE_AA)
        a = cv2.GaussianBlur(a, (0, 0), 0.7)
        m = (a.astype(np.float32) / 255.0)[..., None]
        if len(_mask_cache) < 64:
            _mask_cache[key] = m
    return m


def fill_rounded(img, x0, y0, x1, y1, r, color):
    r = max(0, min(int(r), (x1 - x0) // 2, (y1 - y0) // 2))
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1, cv2.LINE_AA)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r),
                   (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def panel(frame, x0, y0, x1, y1, r, color=TINT, alpha=0.5):
    """Полупрозрачная скруглённая панель — быстрый путь через addWeighted.
    Снаружи скругления overlay == ROI, поэтому те пиксели не меняются."""
    x0, y0, x1, y1 = _clamp(frame, x0, y0, x1, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return
    roi = frame[y0:y1, x0:x1]
    ov = roi.copy()
    fill_rounded(ov, 0, 0, roi.shape[1] - 1, roi.shape[0] - 1, r, color)
    cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, dst=roi)


def paste_clipped(dst, img, x, y):
    H, W = dst.shape[:2]
    ih, iw = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + iw), min(H, y + ih)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = img[y0 - y:y1 - y, x0 - x:x1 - x]


def paste_rounded(dst, img, x, y, radius):
    ih, iw = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + iw), min(dst.shape[0], y + ih)
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0 - y:y1 - y, x0 - x:x1 - x]
    m = rounded_mask(sub.shape[0], sub.shape[1], radius)
    region = dst[y0:y1, x0:x1]
    region[:] = (sub * m + region * (1 - m)).astype(np.uint8)


def fit_into(img, box_w, box_h):
    ph, pw = img.shape[:2]
    s = min(box_w / pw, box_h / ph)
    return cv2.resize(img, (max(1, int(pw * s)), max(1, int(ph * s))))


def dim_full(frame, dark):
    return (frame * (1 - dark)).astype(np.uint8)


def draw_corners(frame, rect, color, thick):
    x0, y0, x1, y1 = rect
    L = max(14, int(min(x1 - x0, y1 - y0) * 0.10))
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                             (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + sx * L, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + sy * L), color, thick, cv2.LINE_AA)


def pill(frame, texts, cx, cy_bottom, text, color, S):
    px = int(22 * S)
    tw = text_w(text, px, "Medium")
    padx, pady = int(20 * S), int(11 * S)
    pw, ph = tw + 2 * padx, int(px * 1.15) + 2 * pady
    x0 = cx - pw // 2
    y0 = cy_bottom - ph
    panel(frame, x0, y0, x0 + pw, cy_bottom, ph // 2, alpha=0.55)
    T(texts, text, 0, y0 + pady, px, color, "cx", weight="Medium")


def draw_cursor(frame, cx, cy, pinch, S):
    if pinch:
        cv2.circle(frame, (cx, cy), int(9 * S), COL_ACCENT, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(4 * S), COL_ACCENT, -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (cx, cy), int(13 * S), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(3 * S), (255, 255, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Жесты
# ---------------------------------------------------------------------------
def landmarks_to_px(hand_landmarks, w, h):
    return np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark],
                    dtype=np.float32)


def hand_size(pts):
    return float(np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP]))


def pinch_ratio(pts):
    size = hand_size(pts)
    if size < 1e-3:
        return 999.0
    return float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / size)


def pinch_point(pts):
    return (pts[THUMB_TIP] + pts[INDEX_TIP]) * 0.5


def framing_rect(hands_pts, w, h):
    if len(hands_pts) < 2:
        return None
    tips = np.array([pts[t] for pts in hands_pts for t in FRAME_TIPS],
                    dtype=np.float32)
    pad = FRAME_PAD * min(w, h)
    x0 = max(0, int(tips[:, 0].min() - pad))
    y0 = max(0, int(tips[:, 1].min() - pad))
    x1 = min(w, int(tips[:, 0].max() + pad))
    y1 = min(h, int(tips[:, 1].max() + pad))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)


def rect_big_enough(rect, w, h):
    x0, y0, x1, y1 = rect
    return (x1 - x0) >= MIN_FRAME_FRAC * w and (y1 - y0) >= MIN_FRAME_FRAC * h


# ---------------------------------------------------------------------------
# Пазл
# ---------------------------------------------------------------------------
def make_board(shot, area, n):
    ax0, ay0, ax1, ay1 = area
    aw, ah = ax1 - ax0, ay1 - ay0
    sh, sw = shot.shape[:2]
    aspect = sw / sh
    bw = aw * BOARD_FILL
    bh = bw / aspect
    if bh > ah * BOARD_FILL:
        bh = ah * BOARD_FILL
        bw = bh * aspect
    cell_w = int((bw - (n + 1) * CELL_GAP) / n)
    cell_h = int((bh - (n + 1) * CELL_GAP) / n)
    total_w = cell_w * n + (n + 1) * CELL_GAP
    total_h = cell_h * n + (n + 1) * CELL_GAP
    bx = ax0 + (aw - total_w) // 2
    by = ay0 + (ah - total_h) // 2

    tiles = []
    for r in range(n):
        for c in range(n):
            y0, y1 = r * sh // n, (r + 1) * sh // n
            x0, x1 = c * sw // n, (c + 1) * sw // n
            tiles.append(cv2.resize(shot[y0:y1, x0:x1], (cell_w, cell_h)))
    tiles_dim = [(t * 0.72).astype(np.uint8) for t in tiles]

    order = list(range(n * n))
    while order == list(range(n * n)):
        random.shuffle(order)

    return {"tiles": tiles, "dim": tiles_dim, "order": order, "held": None,
            "n": n, "bx": bx, "by": by, "cell_w": cell_w, "cell_h": cell_h,
            "total_w": total_w, "total_h": total_h}


def cell_origin(board, pos):
    n = board["n"]
    r, c = divmod(pos, n)
    x = board["bx"] + CELL_GAP + c * (board["cell_w"] + CELL_GAP)
    y = board["by"] + CELL_GAP + r * (board["cell_h"] + CELL_GAP)
    return x, y


def board_bbox(board):
    return (board["bx"], board["by"],
            board["bx"] + board["total_w"], board["by"] + board["total_h"])


def point_to_cell(board, px, py):
    """Ближайшая клетка (прощающее попадание). None — если курсор вне доски."""
    bx0, by0, bx1, by1 = board_bbox(board)
    margin = board["cell_w"] // 2
    if not (bx0 - margin <= px <= bx1 + margin and by0 - margin <= py <= by1 + margin):
        return None
    n = board["n"]
    step_x = board["cell_w"] + CELL_GAP
    step_y = board["cell_h"] + CELL_GAP
    c = int((px - (bx0 + CELL_GAP)) // step_x)
    r = int((py - (by0 + CELL_GAP)) // step_y)
    c = max(0, min(n - 1, c))
    r = max(0, min(n - 1, r))
    return r * n + c


def is_solved(board):
    return board["order"] == list(range(board["n"] * board["n"]))


def draw_board(frame, board, S, active=None, all_bright=False):
    bx0, by0, bx1, by1 = board_bbox(board)
    pad = int(14 * S)
    panel(frame, bx0 - pad, by0 - pad, bx1 + pad, by1 + pad, int(20 * S),
          alpha=0.42)
    n = board["n"]
    for pos in range(n * n):
        if board["held"] == pos:
            continue
        x, y = cell_origin(board, pos)
        src = board["tiles"] if (all_bright or pos == active) else board["dim"]
        paste_clipped(frame, src[board["order"][pos]], x, y)


# ---------------------------------------------------------------------------
# Лента
# ---------------------------------------------------------------------------
def build_strip(photos):
    inner = STRIP_W - 2 * STRIP_PAD
    resized = [cv2.resize(p, (inner, max(1, int(inner * p.shape[0] / p.shape[1]))))
               for p in photos]
    body_h = STRIP_PAD * 2 + sum(r.shape[0] for r in resized) \
        + STRIP_GAP * (len(resized) - 1)
    footer_h = 72
    strip = np.full((body_h + footer_h, STRIP_W, 3), STRIP_BG, np.uint8)
    y = STRIP_PAD
    for r in resized:
        strip[y:y + r.shape[0], STRIP_PAD:STRIP_PAD + inner] = r
        cv2.rectangle(strip, (STRIP_PAD, y), (STRIP_PAD + inner, y + r.shape[0]),
                      (214, 209, 204), 1, cv2.LINE_AA)
        y += r.shape[0] + STRIP_GAP
    img = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    font = get_font(27, "Semibold", rounded=True)
    cap = "Photo Puzzle Booth  ·  " + datetime.now().strftime("%d.%m.%Y")
    bb = d.textbbox((0, 0), cap, font=font)
    d.text(((STRIP_W - (bb[2] - bb[0])) // 2, body_h + 22), cap,
           font=font, fill=(70, 70, 75))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def save_image(img, prefix, directory=OUTPUT_DIR):
    os.makedirs(directory, exist_ok=True)
    name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.png")
    path = os.path.join(directory, name)
    cv2.imwrite(path, img)
    print("Сохранено:", path)
    return path


def sidebar_rect(w, h, S):
    m = int(20 * S)
    pw = int(max(240 * S, w * 0.22))
    return (w - m - pw, m, w - m, h - m)


def draw_sidebar(frame, texts, collected, S):
    h, w = frame.shape[:2]
    sx0, sy0, sx1, sy1 = sidebar_rect(w, h, S)
    panel(frame, sx0, sy0, sx1, sy1, int(24 * S), alpha=0.5)

    pad = int(20 * S)
    T(texts, "Фотобудка", sx0 + pad, sy0 + int(18 * S), int(28 * S),
      COL_TEXT, weight="Semibold", rounded=True)
    T(texts, f"{len(collected)} / {NUM_SHOTS}", sx0 + pad, sy0 + int(52 * S),
      int(17 * S), COL_ACCENT, weight="Semibold")

    top = sy0 + int(86 * S)
    bottom = sy1 - int(14 * S)
    gap = int(12 * S)
    slot_h = (bottom - top - (NUM_SHOTS - 1) * gap) // NUM_SHOTS
    for i in range(NUM_SHOTS):
        y0 = top + i * (slot_h + gap)
        x0, x1, y1 = sx0 + pad, sx1 - pad, y0 + slot_h
        filled = i < len(collected)
        panel(frame, x0, y0, x1, y1, int(14 * S),
              alpha=0.28 if filled else 0.16)
        if filled:
            inset = int(9 * S)
            thumb = fit_into(collected[i], x1 - x0 - 2 * inset,
                             y1 - y0 - 2 * inset)
            ox = x0 + (x1 - x0 - thumb.shape[1]) // 2
            oy = y0 + (y1 - y0 - thumb.shape[0]) // 2
            paste_clipped(frame, thumb, ox, oy)
        else:
            npx = int(26 * S)
            tw = text_w(str(i + 1), npx, "Light")
            T(texts, str(i + 1), x0 + (x1 - x0 - tw) // 2,
              y0 + slot_h // 2 - npx // 2, npx, (130, 118, 112), weight="Light")


# ---------------------------------------------------------------------------
def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    return cap


def main() -> None:
    cap = open_camera()
    if not cap.isOpened():
        print(f"Не удалось открыть камеру index={CAMERA_INDEX}.")
        return

    hands = mp_hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    prev_t = time.time()
    fps = 0.0
    pinch_state = {"Left": False, "Right": False}
    ratio_ema = {"Left": None, "Right": None}
    last_rect = None
    rect_ema = None
    both_pinch_prev = False
    puzzle_pinch_prev = False
    cursor_ema = None
    last_hover = None

    state = "FRAME"
    locked_rect = None
    countdown_until = review_until = flash_until = solved_until = 0.0
    review_img = None
    current_shot = None
    board = None
    collected = []
    strip_img = None
    win_sized = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        S = h / 720.0
        clean = frame.copy()
        now = time.time()
        texts = []

        scale = PROCESS_WIDTH / w
        small = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)

        hands_pts, hands_pinch = [], []
        if result.multi_hand_landmarks:
            handed = result.multi_handedness or []
            used = []
            for i, hlm in enumerate(result.multi_hand_landmarks):
                if state in ("FRAME", "PUZZLE"):
                    mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS,
                                           LM_SPEC, CONN_SPEC)
                pts = landmarks_to_px(hlm, w, h)
                hands_pts.append(pts)
                label = handed[i].classification[0].label if i < len(handed) else "Left"
                if label in used:
                    label = "Right" if "Right" not in used else "Left"
                used.append(label)
                ratio = pinch_ratio(pts)
                prev_r = ratio_ema.get(label)
                ratio = ratio if prev_r is None else (
                    RATIO_EMA * ratio + (1 - RATIO_EMA) * prev_r)
                ratio_ema[label] = ratio
                if pinch_state.get(label, False):
                    if ratio > PINCH_OFF:
                        pinch_state[label] = False
                else:
                    if ratio < PINCH_ON:
                        pinch_state[label] = True
                hands_pinch.append(pinch_state.get(label, False))

        two_hands = len(hands_pts) >= 2
        both_pinch = two_hands and all(hands_pinch[:2])

        # ===================================================================
        if state == "FRAME":
            if two_hands and not both_pinch:
                r = framing_rect(hands_pts, w, h)
                if r is not None:
                    if rect_ema is None:
                        rect_ema = np.array(r, np.float32)
                    else:
                        rect_ema = (RECT_EMA * np.array(r, np.float32)
                                    + (1 - RECT_EMA) * rect_ema)
                    last_rect = tuple(int(v) for v in rect_ema)
            elif not two_hands:
                rect_ema = None
            if (both_pinch and not both_pinch_prev and last_rect is not None
                    and rect_big_enough(last_rect, w, h)):
                locked_rect = last_rect
                countdown_until = now + COUNTDOWN_SEC
                state = "COUNTDOWN"
            if last_rect is not None and two_hands:
                oks = rect_big_enough(last_rect, w, h)
                col = COL_SUCCESS if oks else COL_WARN
                draw_corners(frame, last_rect, col, max(1, int(2 * S)))
                pill(frame, texts, w // 2, h - int(26 * S),
                     "сведи оба щипка" if oks else "разведи руки шире", col, S)
            elif not two_hands:
                pill(frame, texts, w // 2, h - int(26 * S),
                     "покажи две руки и обведи область", COL_TEXT, S)

        elif state == "COUNTDOWN":
            frame = dim_full(frame, 0.32)
            draw_corners(frame, locked_rect, COL_SUCCESS, max(1, int(2 * S)))
            remain = countdown_until - now
            if remain <= 0:
                x0, y0, x1, y1 = locked_rect
                current_shot = clean[y0:y1, x0:x1].copy()
                save_image(current_shot, "shot")
                review_img = current_shot
                review_until = now + REVIEW_SEC
                flash_until = now + FLASH_SEC
                state = "REVIEW"
            else:
                T(texts, str(int(np.ceil(remain))), 0, 0, int(190 * S),
                  COL_TEXT, "center", -int(10 * S), weight="Thin")
                pill(frame, texts, w // 2, h - int(26 * S), "позируй",
                     COL_SUCCESS, S)

        elif state == "REVIEW":
            frame = dim_full(frame, 0.5)
            disp = fit_into(review_img, int(w * 0.5), int(h * 0.6))
            ox, oy = (w - disp.shape[1]) // 2, (h - disp.shape[0]) // 2
            paste_rounded(frame, disp, ox, oy, int(14 * S))
            if now >= review_until:
                area = (int(20 * S), int(20 * S),
                        sidebar_rect(w, h, S)[0] - int(12 * S), h - int(20 * S))
                board = make_board(review_img, area, PUZZLE_N)
                cursor_ema = None
                last_hover = None
                state = "PUZZLE"

        elif state == "PUZZLE":
            cursor, ctrl_pinch = None, False
            if hands_pts:
                ci = 0
                if len(hands_pts) > 1:
                    p0, p1 = pinch_point(hands_pts[0]), pinch_point(hands_pts[1])
                    if cursor_ema is not None:
                        ci = 0 if (np.linalg.norm(p0 - cursor_ema)
                                   <= np.linalg.norm(p1 - cursor_ema)) else 1
                    if hands_pinch[0] != hands_pinch[1]:
                        ci = 0 if hands_pinch[0] else 1
                raw = pinch_point(hands_pts[ci])
                cursor_ema = raw if cursor_ema is None else (
                    CURSOR_EMA * raw + (1 - CURSOR_EMA) * cursor_ema)
                cursor = cursor_ema
                ctrl_pinch = hands_pinch[ci]
            else:
                cursor_ema = None

            hover = point_to_cell(board, *cursor) if cursor is not None else None
            draw_board(frame, board, S, active=hover)

            if cursor is not None:
                cx, cy = int(cursor[0]), int(cursor[1])
                if ctrl_pinch and not puzzle_pinch_prev:        # взять
                    if hover is not None:
                        board["held"] = hover
                        last_hover = hover
                elif ctrl_pinch and board["held"] is not None:  # держим
                    if hover is not None:
                        last_hover = hover
                elif (not ctrl_pinch and puzzle_pinch_prev
                      and board["held"] is not None):           # отпустили
                    if last_hover is not None and last_hover != board["held"]:
                        o = board["order"]
                        o[board["held"]], o[last_hover] = o[last_hover], o[board["held"]]
                    board["held"] = None
                    if is_solved(board):
                        collected.append(current_shot)
                        solved_until = now + SOLVED_SEC
                        state = "SOLVED"
                if board["held"] is not None:
                    tile = board["tiles"][board["order"][board["held"]]]
                    paste_clipped(frame, tile, cx - board["cell_w"] // 2,
                                  cy - board["cell_h"] // 2)
                draw_cursor(frame, cx, cy, ctrl_pinch, S)
            puzzle_pinch_prev = ctrl_pinch
            bx0, _, bx1, _ = board_bbox(board)
            pill(frame, texts, (bx0 + bx1) // 2, h - int(26 * S),
                 "щипком переставляй куски", COL_DIM, S)

        elif state == "SOLVED":
            if board is not None:
                draw_board(frame, board, S, all_bright=True)
                bx0, by0, bx1, _ = board_bbox(board)
                tw = text_w("Собрано", int(52 * S), "Bold", rounded=True)
                T(texts, "Собрано", bx0 + (bx1 - bx0 - tw) // 2,
                  by0 - int(72 * S), int(52 * S), COL_SUCCESS,
                  weight="Bold", rounded=True)
            if now >= solved_until:
                board = None
                last_rect = None
                rect_ema = None
                if len(collected) >= NUM_SHOTS:
                    strip_img = build_strip(collected)
                    save_image(strip_img, "strip", DESKTOP_DIR)
                    state = "DONE"
                else:
                    state = "FRAME"

        elif state == "DONE":
            frame = dim_full(frame, 0.58)
            disp = fit_into(strip_img, int(w * 0.3), int(h * 0.82))
            ox, oy = (w - disp.shape[1]) // 2, (h - disp.shape[0]) // 2 + int(14 * S)
            paste_rounded(frame, disp, ox, oy, int(10 * S))
            tw = text_w("Лента готова", int(38 * S), "Semibold", rounded=True)
            T(texts, "Лента готова", (w - tw) // 2, int(24 * S), int(38 * S),
              COL_TEXT, weight="Semibold", rounded=True)
            pill(frame, texts, w // 2, h - int(24 * S),
                 "сохранено на Рабочий стол   ·   r — заново", COL_TEXT, S)

        if now < flash_until:
            a = (flash_until - now) / FLASH_SEC
            frame = cv2.addWeighted(np.full_like(frame, 255), a, frame, 1 - a, 0)

        both_pinch_prev = both_pinch

        if state in ("FRAME", "COUNTDOWN", "PUZZLE", "SOLVED"):
            draw_sidebar(frame, texts, collected, S)

        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        T(texts, f"{fps:4.0f} fps", int(16 * S), int(12 * S), int(15 * S),
          COL_DIM)
        T(texts, "q — выход", int(16 * S), h - int(28 * S), int(15 * S), COL_DIM)

        frame = render_texts(frame, texts, w, h)

        if not win_sized:
            sc = min(1600 / w, 900 / h, 1.0)
            cv2.resizeWindow(WINDOW_NAME, int(w * sc), int(h * sc))
            win_sized = True

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r") and state == "DONE":
            collected = []
            strip_img = None
            last_rect = None
            rect_ema = None
            state = "FRAME"

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
