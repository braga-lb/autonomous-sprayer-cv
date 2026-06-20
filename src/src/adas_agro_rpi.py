import cv2
import numpy as np
from collections import deque
from picamera2 import Picamera2

# Habilita otimizações nativas SIMD/NEON do OpenCV
cv2.setUseOptimized(True)

# --- CONFIGURAÇÕES OTIMIZADAS PARA RPI + PICAMERA2 ---
# Resolução reduzida (75% menos processamento vs 640x480)
W, H = 320, 240

LIMITE_PIXELS_PLANTA = 375       # Proporcional a 320x240
AREA_MINIMA_PARA_NAVEGAR = 1250
LARGURA_PULVERIZADOR = 120
COR_PULVERIZADOR = (50, 180, 255)
BUFFER_CURVA = 15

historico_curva = deque(maxlen=BUFFER_CURVA)

# MOG2 otimizado: shadows=False economiza CPU no Raspberry
backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=68, detectShadows=False)

# Kernels pré-alocados (evita recriação no loop)
kernel_limpeza = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_dilate  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def process_frame(frame):
    if frame is None: return None

    # ---------------------------------------------------------
    # PARTE A: DETECÇÃO DE OBSTÁCULOS
    # ---------------------------------------------------------
    y_start, y_end = int(H * 0.1), int(H * 0.95)
    x_start, x_end = int(W * 0.15), int(W * 0.85)
    roi_seguranca = frame[y_start:y_end, x_start:x_end]

    fg_mask = backSub.apply(roi_seguranca)
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_limpeza)

    contours_obs, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstaculo_detectado = False
    for cnt in contours_obs:
        area = cv2.contourArea(cnt)
        if area > 212:
            x_obs, y_obs, w_obs, h_obs = cv2.boundingRect(cnt)
            proporcao = h_obs / float(w_obs)
            if proporcao > 0.42:
                obstaculo_detectado = True
                cv2.rectangle(frame, (x_obs + x_start, y_obs + y_start),
                              (x_obs + x_start + w_obs, y_obs + y_start + h_obs), (0, 0, 255), 2)
                break  # Para no primeiro obstáculo para poupar CPU

    # ---------------------------------------------------------
    # PARTE B: NAVEGAÇÃO — SEGMENTAÇÃO LAB
    # ---------------------------------------------------------
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a_chan = cv2.extractChannel(lab, 1)  # extractChannel evita o pesado split()

    _, mask_bruta = cv2.threshold(a_chan, 118, 255, cv2.THRESH_BINARY_INV)

    contornos_plantas, _ = cv2.findContours(mask_bruta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_fileiras = np.zeros_like(mask_bruta)
    area_total_plantas = 0
    for cnt in contornos_plantas:
        area = cv2.contourArea(cnt)
        if area > LIMITE_PIXELS_PLANTA:
            cv2.drawContours(mask_fileiras, [cnt], -1, 255, thickness=cv2.FILLED)
            area_total_plantas += area

    mask_fileiras_dilatada = cv2.dilate(mask_fileiras, kernel_dilate)
    mask_caminho_livre = cv2.bitwise_not(mask_fileiras_dilatada)

    # Distance Transform com maskSize=3 (mais leve no Pi)
    dist_raw = cv2.distanceTransform(mask_caminho_livre, cv2.DIST_L2, 3)

    alturas = [int(H * 0.45), int(H * 0.65), int(H * 0.9)]
    pontos_x = []

    if area_total_plantas < AREA_MINIMA_PARA_NAVEGAR:
        pontos_x = [W // 2, W // 2, W // 2]
    else:
        for h in alturas:
            fatia = dist_raw[h - 5:h + 5, :]
            soma_fatia = np.sum(fatia, axis=0)
            pontos_x.append(int(np.argmax(soma_fatia)) if np.max(soma_fatia) > 0 else W // 2)

    historico_curva.append(pontos_x)
    avg_pts = np.mean(historico_curva, axis=0).astype(int)

    # ---------------------------------------------------------
    # PARTE C: RENDERIZAÇÃO (inteiros para economizar CPU)
    # ---------------------------------------------------------
    overlay = frame.copy()
    pts_esq, pts_dir = [], []
    for i, h in enumerate(alturas):
        larg = LARGURA_PULVERIZADOR if i > 0 else int(LARGURA_PULVERIZADOR * 0.7)
        pts_esq.append([avg_pts[i] - larg // 2, h])
        pts_dir.insert(0, [avg_pts[i] + larg // 2, h])

    poly_pts = np.array(pts_esq + pts_dir, np.int32)
    cv2.fillPoly(overlay, [poly_pts], COR_PULVERIZADOR)
    cv2.drawContours(overlay, contornos_plantas, -1, (0, 255, 255), 1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # ---------------------------------------------------------
    # PARTE D: HUD
    # ---------------------------------------------------------
    erro = avg_pts[2] - W // 2
    cv2.rectangle(frame, (W // 2 - 1, int(H * 0.2)), (W // 2 + 1, H), (0, 0, 255), -1)
    cv2.rectangle(frame, (5, 5), (W - 5, 45), (0, 0, 0), -1)

    if obstaculo_detectado:
        msg, cor = "FREIO DE EMERGENCIA", (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (W, 10), cor, -1)
    elif area_total_plantas < AREA_MINIMA_PARA_NAVEGAR:
        msg, cor = "CENTRAL - SEM PLANTA", (255, 255, 255)
    elif abs(erro) < 10:
        msg, cor = "VIA LIVRE", (0, 255, 0)
    else:
        msg, cor = f"AJUSTAR: {'DIR' if erro > 0 else 'ESQ'}", (0, 200, 255)

    cv2.putText(frame, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA)
    cv2.putText(frame, f"DESVIO: {erro}px | OBS: {'SIM' if obstaculo_detectado else 'NAO'}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# =========================================================
# EXECUÇÃO — PICAMERA2 (Hardware nativo do Raspberry Pi)
# =========================================================
print("Inicializando câmera nativa (Picamera2)...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (W, H)})
picam2.configure(config)
picam2.start()
print("SISTEMA OPERACIONAL — ADAS AGRO ATIVO")

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_cv  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        result    = process_frame(frame_cv)
        if result is not None:
            cv2.imshow("ADAS Agro — Picamera2 RPi", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nParando o sistema...")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
