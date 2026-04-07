# Final Year Project @ ISE
## $\color{purple}{\textsf{Low-Power Embedded Bird-Call Classification for Irish Species Monitoring }}$ 🦢

#### Abstract
Automated bird-audio classification is increasingly
important for biodiversity monitoring, but many current systems
rely on cloud inference, large global models, or high-power edge
hardware that is difficult to deploy at scale in field settings.
This paper presents MicroBird, a region-specific embedded
bird-call classification system for Irish species monitoring on
the MAX78002 microcontroller. A reproducible dataset creation
pipeline was developed by deriving a local species list from eBird
occurrence data, collecting focal recordings from Xeno-Canto,
extracting three second clips, and using BirdNET as a teacher
model to filter weak recording- level labels. The resulting clips
were converted into 64×128 log-mel spectrograms and used to
train a compact hardware-aware convolutional neural network
for 51 classes (50 Irish bird species plus a non-bird background
class). The final model was synthesised for the MAX78002 and
flashed to the device. The on-device inference pipeline computes
spectrograms on the Cortex-M4 and performs classification on
the MAX78002 CNN accelerator. The deployed system achieved
79.10% top-1 and 90.90% top-3 accuracy on the full held-out test
set. On-device measurement showed CNN inference completed in
3.5 ms while consuming 15.3 μJ, and the full inference pipeline
required 218.8 ms and 2.168 mJ per 3-second clip.
This is the first reported bird-audio classification system to
achieve microjoule-scale CNN inference in a deployed multi-class
setting. These results show that strong region-specific bird-audio
classification can be achieved in an ultra-low-power embedded
system.
