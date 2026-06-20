# 🌱 Autonomous Sprayer — Computer Vision for Agricultural Vehicles

> Sistema de visão computacional para navegação autônoma de pulverizador em fileiras de cultivo.  
> Desenvolvido como projeto de Iniciação Científica na **UFERSA** (Universidade Federal Rural do Semi-Árido).

![Demo](demo_frame.png)

---

## 🎯 O Problema

Pequenos agricultores realizam a pulverização de defensivos agrícolas de forma manual, expondo-se a agentes químicos nocivos e gerando desperdício por aplicação imprecisa. Este projeto desenvolve um sistema de visão embarcado que permite ao veículo navegar autonomamente entre as fileiras de cultivo.

---

## ⚙️ Como Funciona

O pipeline de processamento roda em tempo real e é dividido em quatro módulos:

**A — Detecção de Obstáculos**  
Subtração de fundo (MOG2) na ROI frontal do veículo. Contornos com proporção vertical > 0.42 acionam parada de emergência.

**B — Segmentação de Fileiras**  
Conversão para espaço de cor LAB + threshold no canal A (sensível a verde vegetal). Distance Transform identifica o corredor central livre entre as fileiras.

**C — Cálculo de Trajetória**  
Análise do corredor em 3 alturas do frame. Histórico de 15 frames suaviza a curva e evita oscilações bruscas.

**D — HUD de Navegação**  
Overlay visual com comando direcional (VIA LIVRE / VIRE DIR / VIRE ESQ / FREIO DE EMERGÊNCIA) e desvio em pixels em relação ao centro.

---

## 🛠️ Stack Técnica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| Visão Computacional | OpenCV 4.x |
| Hardware | Raspberry Pi 4B |
| Câmera | Raspberry Pi Camera Module (Picamera2) |
| Segmentação | Espaço de cor LAB + Distance Transform |
| Detecção de Obstáculos | MOG2 Background Subtraction |

---

## 🚀 Como Rodar

**No PC (simulação com vídeo):**
```bash
pip install -r requirements.txt
# Coloque seu vídeo de teste como video_teste.mp4 na raiz
python src/adas_agro_pc.py
```

**No Raspberry Pi (câmera real):**
```bash
sudo apt install python3-picamera2
pip install opencv-python numpy
python src/adas_agro_rpi.py
```

Pressione `Q` para encerrar.

---

## 📁 Estrutura

```
autonomous-sprayer-cv/
├── src/
│   ├── adas_agro_pc.py     # Versão para teste em PC (vídeo .mp4)
│   └── adas_agro_rpi.py    # Versão otimizada para Raspberry Pi
├── requirements.txt
├── demo_frame.png
└── README.md
```

---

## 🔬 Contexto Acadêmico

Este sistema é parte da pesquisa *"Pulverizador Autônomo: A Tecnologia a Serviço da Saúde do Pequeno Agricultor"*, aceita para apresentação de pôster no **CONBEA 2026** e no **X FECICEEP**.

O projeto gerou um **depósito de patente e desenho industrial no INPI**.

---

## 👤 Autor

**Tallysson Levy Braga**  
Engenharia de Software — UFERSA  
Pesquisador IC — Visão Computacional para Veículos Agrícolas Autônomos  
Fundador — [Ochii Ecosemitec](https://linkedin.com/in/seu-perfil)
