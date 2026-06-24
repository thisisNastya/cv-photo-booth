"""Пробник камер: перебирает индексы 0..3, для каждой пробует получить кадр
и печатает размер + среднюю яркость (чтобы отличить чёрный кадр от реального).
Запускать из Terminal:
    ./.venv/bin/python camera_check.py
"""
import time
import cv2


def probe(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    frame = None
    for _ in range(30):              # дать камере прогреться
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        time.sleep(0.03)
    cap.release()
    if frame is None:
        return (index, None, None)
    mean = float(frame.mean())
    return (index, frame.shape, mean)


def main() -> None:
    print("Перебираю камеры 0..3 (на каждую ~1 сек)...\n")
    found = []
    for i in range(4):
        res = probe(i)
        if res is None:
            print(f"  index {i}: не открылась")
            continue
        idx, shape, mean = res
        if shape is None:
            print(f"  index {i}: открылась, но кадра нет")
        else:
            kind = "ЧЁРНЫЙ кадр" if mean < 8 else "есть картинка"
            print(f"  index {i}: {shape}  ср.яркость={mean:5.1f}  -> {kind}")
            found.append((idx, mean))

    print()
    live = [i for i, m in found if m >= 8]
    if live:
        print(f"Рабочая камера (нормальная картинка): index = {live[0]}")
    elif found:
        print("Камеры открываются, но все кадры чёрные. "
              "Если активна Continuity Camera — телефон должен быть разблокирован "
              "и стоять как камера; или отключи его, чтобы взялась встроенная.")
    else:
        print("Ни одна камера не открылась.")


if __name__ == "__main__":
    main()
