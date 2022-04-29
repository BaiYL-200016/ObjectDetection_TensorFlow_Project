conda create -p=%cd%\tensorflow2 python=3.8 -y && conda activate %CD%\tensorflow2 && conda install tensorflow-gpu=2.6.0 -y && pip install -r requirements.txt && python -m ipykernel install --user --name=tensoflow2
pause

