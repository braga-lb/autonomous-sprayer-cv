import cv2
import numpy as np
from collections import deque

# --- CONFIGURAÇÕES DE ENGENHARIA ---
VIDEO_PATH = "video_teste.mp4"
W, H = 640, 480
LIMITE_PIXELS_PLANTA = 1500
AREA_MINIMA_PARA_NAVEGAR = 5000

# Configurações do Implemento (Pulverizador)
LARGURA_PULVERIZADOR = 240
COR_PULVERIZADOR = (255, 180, 50)  # Azul/Ciano
BUFFER_CURVA = 15

historico_curva = deque(maxlen=BUFFER_CURVA)

# --- CONFIGURAÇÕES DE DETECÇÃO DE OBSTÁCULOS ---
backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=68, detectShadows=True)
kernel_limpeza = np.ones((5, 5), np.uint8)

def process_frame(frame):
    if frame is None: return None
    frame = cv2.resize(frame, (W, H))

    # ---------------------------------------------------------
    # PARTE A: DETECÇÃO DE OBSTÁCULOS (MOG2 Background Subtraction)
    # ---------------------------------------------------------
    y_start, y_end = int(H * 0.1), int(H * 0.95)
    x_start, x_end = int(W * 0.15), int(W * 0.85)
    roi_seguranca = frame[y_start:y_end, x_start:x_end]

    fg_mask = backSub.apply(roi_seguranca)
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_limpeza)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_limpeza)

    contours_obs, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstaculo_detectado = False
    for cnt in contours_obs:
        area = cv2.contourArea(cnt)
        x_obs, y_obs, w_obs, h_obs = cv2.boundingRect(cnt)
        proporcao = h_obs / float(w_obs)

        if area > 850 and proporcao > 0.42:
            obstaculo_detectado = True
            cv2.rectangle(frame, (x_obs + x_start, y_obs + y_start),
                          (x_obs + x_start + w_obs, y_obs + y_start + h_obs), (0, 0, 255), 3)

    # ---------------------------------------------------------
    # PARTE B: NAVEGAÇÃO — SEGMENTAÇÃO DE FILEIRAS (Espaço de cor LAB)
    # ---------------------------------------------------------
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    _, a_chan, _ = cv2.split(lab)
    _, mask_bruta = cv2.threshold(a_chan, 118, 255, cv2.THRESH_BINARY_INV)

    # Remove detecções falsas no centro (solo claro)
    mask_bruta[:, 180:460] = 0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bruta, connectivity=8)
    mask_fileiras = np.zeros_like(mask_bruta)
    area_total_plantas = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > LIMITE_PIXELS_PLANTA:
            mask_fileiras[labels == i] = 255
            area_total_plantas += area

    mask_fileiras = cv2.morphologyEx(mask_fileiras, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Distance Transform: encontra o corredor central livre
    mask_caminho_livre = cv2.bitwise_not(cv2.dilate(mask_fileiras, np.ones((10, 10), np.uint8)))
    dist_raw = cv2.distanceTransform(mask_caminho_livre, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_raw.copy(), None, 0, 1.0, cv2.NORM_MINMAX)

    # Trajetória suavizada em 3 alturas
    alturas = [int(H * 0.45), int(H * 0.65), int(H * 0.9)]
    pontos_x = []

    if area_total_plantas < AREA_MINIMA_PARA_NAVEGAR:
        pontos_x = [W // 2, W // 2, W // 2]
    else:
        for h in alturas:
            fatia = dist_raw[h - 10:h + 10, :]
            soma_fatia = np.sum(fatia, axis=0)
            pontos_x.append(np.argmax(soma_fatia) if np.max(soma_fatia) > 0 else W // 2)

    historico_curva.append(pontos_x)
    avg_pts = np.mean(historico_curva, axis=0).astype(int)

    # ---------------------------------------------------------
    # PARTE C: RENDERIZAÇÃO VISUAL
    # ---------------------------------------------------------
    img_f = frame.astype(np.float32) / 255.0
    grad_v = np.tile(np.linspace(0, 1, H), (W, 1)).T

    _, faixa_v_bin = cv2.threshold(dist_norm, 0.1, 1.0, cv2.THRESH_BINARY)
    faixa_v_bin = (faixa_v_bin * 255).astype(np.uint8)
    mask_alpha_v = cv2.GaussianBlur(faixa_v_bin, (35, 35), 15).astype(np.float32) / 255.0
    alpha_v_3ch = cv2.merge([mask_alpha_v * grad_v] * 3)
    paint_v = np.zeros_like(frame)
    paint_v[faixa_v_bin > 0] = [0, 255, 100]
    res_f = (img_f * (1.0 - alpha_v_3ch)) + (img_f * (paint_v.astype(np.float32) / 255.0) * 3.5 * alpha_v_3ch)

    mask_pulv = np.zeros((H, W), dtype=np.uint8)
    pts_esq, pts_dir = [], []
    for i, h in enumerate(alturas):
        larg = LARGURA_PULVERIZADOR if i > 0 else int(LARGURA_PULVERIZADOR * 0.7)
        pts_esq.append([avg_pts[i] - larg // 2, h])
        pts_dir.insert(0, [avg_pts[i] + larg // 2, h])
    cv2.fillPoly(mask_pulv, [np.array(pts_esq + pts_dir, np.int32)], 255)
    mask_alpha_p = (cv2.GaussianBlur(mask_pulv, (21, 21), 10).astype(np.float32) / 255.0) * grad_v * 0.75
    res_f = (res_f * (1.0 - cv2.merge([mask_alpha_p] * 3))) + (
            np.array(COR_PULVERIZADOR) / 255.0 * cv2.merge([mask_alpha_p] * 3))

    mask_solo = cv2.bitwise_not(cv2.bitwise_or(mask_fileiras, faixa_v_bin))
    solo_3ch = cv2.merge([mask_solo.astype(np.float32) / 255.0] * 3)
    solo_roxo = res_f * (np.array([0.5, 0.1, 0.4], dtype=np.float32) + 0.5)
    res_f = res_f * (1.0 - solo_3ch) + solo_roxo * solo_3ch

    res = (np.clip(res_f * 255, 0, 255)).astype(np.uint8)

    # ---------------------------------------------------------
    # PARTE D: HUD — COMANDOS DE NAVEGAÇÃO
    # ---------------------------------------------------------
    contornos_p, _ = cv2.findContours(mask_fileiras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, contornos_p, -1, (0, 255, 255), 2)
    cv2.rectangle(res, (W // 2 - 2, 0), (W // 2 + 2, H), (0, 0, 255), -1)

    erro = avg_pts[2] - W // 2
    cv2.rectangle(res, (10, 10), (450, 95), (0, 0, 0), -1)

    if obstaculo_detectado:
        msg, cor = "!!! FREIO DE EMERGENCIA !!!", (0, 0, 255)
    elif area_total_plantas < AREA_MINIMA_PARA_NAVEGAR:
        msg, cor = "CENTRALIZADO (SEM PLANTA)", (255, 255, 255)
    elif abs(erro) < 18:
        msg, cor = "VIA LIVRE: SIGA EM FRENTE", (0, 255, 0)
    else:
        msg, cor = f"AJUSTAR DIRECAO: {'DIR' if erro > 0 else 'ESQ'}", (255, 200, 0)

    cv2.putText(res, msg, (20, 50), 1, 1.4, cor, 2)
    cv2.putText(res, f"DESVIO: {erro}px | OBS: {'SIM' if obstaculo_detectado else 'NAO'}", (20, 80), 1, 0.9,
                (255, 255, 255), 1)

    return res

# --- EXECUÇÃO ---
cap = cv2.VideoCapture(VIDEO_PATH)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    result = process_frame(frame)
    if result is not None:
        cv2.imshow("ADAS Agro + Obstacle Detection", result)
    if cv2.waitKey(20) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
