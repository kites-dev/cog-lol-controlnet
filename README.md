# Stable Diffusion Controlnet v2 Cog model

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt="monkey scuba diving" -i image="image url" -i controlimage="qrcode"
