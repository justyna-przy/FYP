# Final Year Project @ ISE
## $\color{purple}{\textsf{Low-Power Embedded Bird-Call Classification for Irish Species Monitoring }}$ 🦢

#### Abstract
Monitoring bird populations provides valuable insight into biodiversity and climate change, yet large-scale audio monitoring remains expensive and data-intensive. Current solutions, such as BirdNET or Bird@Edge, rely on cloud inference or high-power edge devices, making wide deployment impractical.

This project explores a low-power embedded approach to regional bird-call classification for Irish species using the MAX78000 microcontroller. The project proposes replacing large global models with compact, region-specific models optimized for edge deployment. The model will be trained on log-mel spectrograms of region-specific Irish species using public datasets such as Xeno-canto and the Macaulay Library. Once trained, it will be quantized with Analog Devices’ ai8x toolchain and deployed on nodes consisting of the MAX78000, a microphone, and minimal support hardware. The end goal is to field multiple nodes that send compact, timestamped predictions to a local gateway over a low-power network (e.g., LoRa/LoRaWAN) and to visualise detected species remotely.
