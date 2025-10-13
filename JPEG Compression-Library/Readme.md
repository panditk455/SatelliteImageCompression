
1) Create & activate a virtual environment(so that u dont have to install all in your comp)
easier for everyone!

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip


install:
pip install pillow numpy imagecodecs

Problem:

glymur had a problem with my own interface so I just went with imagecodecs
JPEG 2000 did not work for me.

How to test:

# python3 conversion.py images/airplane.bmp images/inputfile5.tif images/boats.bmp images/goldhill.bmp --outdir output_folder --quality 20

# u can add more files if u wnat to test that out! 