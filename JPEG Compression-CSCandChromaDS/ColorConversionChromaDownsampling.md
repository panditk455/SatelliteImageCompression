Most images start in the RGB color space, where every pixel is described by its red, green, and blue components. These three values mix together to produce all visible colors, just like blending paints. For example, a pixel with values (255, 0, 0) represents pure red, while (100, 100, 100) represents gray. In essence, an RGB image is made up of three separate grayscale layers, one for each color channel, stacked on top of each other.

However, the JPEG format doesn’t work directly with RGB because the human eye doesn’t perceive all aspects of color equally. People are far more sensitive to differences in brightness (light versus dark) than to differences in color. JPEG takes advantage of this by changing the way color is represented in the image. Instead of working with red, green, and blue channels, it transforms the image into a new color space called YCbCr, which separates brightness from color information. This allows JPEG to retain the sharpness and brightness details while compressing color information more aggressively, without a noticeable loss in quality.

In the YCbCr color space, the Y channel represents luminance, or brightness. It captures the lightness or darkness of each pixel and contains most of the visual detail of the image. The Cb channel (blue-difference chroma) measures how blue a pixel is compared to its brightness, and the Cr channel (red-difference chroma) measures how red it is. Together, Cb and Cr represent the color content of the image, while Y represents the structure and lightness.

Mathematically, the conversion from RGB to YCbCr can be expressed using simple linear equations. For example, the brightness channel (Y) is calculated as roughly 0.299*R + 0.587*G + 0.114*B, while the chroma channels are computed as differences between the RGB values and brightness. The Y channel depends most on green values because human vision is most sensitive to green light. When using Python’s Pillow library, this conversion happens automatically when you call "img.convert("YCbCr"). The library applies these formulas to every pixel and produces three separate arrays: one for Y, one for Cb, and one for Cr.

If we save and view these channels individually, we can clearly see their different roles. The Y image looks like a normal black-and-white photograph because it contains the fine details and structure. The Cb and Cr images, on the other hand, look fuzzy or abstract—they show color variation without brightness information.(I have done that in part 2, which is Chroma Downsampling! ) JPEG later compresses these two color channels much more aggressively, because losing small color details has little impact on what humans perceive as image quality.

So, this color conversion  part in the JPEG allows JPEG to separate the parts of an image that matter most to human vision from the parts that can be simplified. By working in the YCbCr color space, the algorithm can preserve brightness and edges (which our eyes notice most) while reducing the size of color data. This step lays the foundation for chroma downsampling and the another part lays foundation for this so that we can preciesely do the other parts too.

Chroma DownSampling:


After converting an image from RGB to YCbCr, the next major step in JPEG compression is chroma downsampling. This process reduces the amount of color information stored in the image, based on the fact that the human eye is far more sensitive to changes in brightness than to small variations in color. In other words, we can remove a lot of the color detail without noticeably affecting how the image looks to us. Chroma downsampling is one of the main reasons JPEG achieves such high compression ratios while still appearing visually similar to the original image.

To understand what’s being "downsampled,", in the YCbCr color space, Y represents the brightness (luminance) of each pixel, while Cb and Cr represent the color components (chrominance). The luminance channel Y carries the edges, textures, and fine details of the image, everything our eyes are very good at detecting. The chrominance channels, Cb and Cr, describe how blue or red each area is, but our eyes can tolerate those values being less precise. JPEG takes advantage of this by storing the color information at a lower resolution, meaning fewer samples are used to describe Cb and Cr compared to Y.

The most common method of chroma downsampling used in JPEG is called 4:2:0 subsampling. The numbers describe how much data is kept for each channel in a small block of pixels. Imagine we start with a block of 4 pixels (2x2). The Y channel keeps all 4 samples, one for every pixel, since brightness is important. However, the Cb and Cr channels each keep only 1 sample for that same 2×2 block. This effectively reduces the color resolution by half in both the horizontal and vertical directions. So, the color data is made coarser, while brightness stays sharp. So, if the original Y channel has a size of 1000×1000 pixels, the Cb and Cr channels after 4:2:0 downsampling might only be 500x500 pixels each.

When JPEG later reconstructs the image, these smaller Cb and Cr channels need to be scaled back up to match the full size of the Y channel. This process is called upsampling, and it’s usually done using simple interpolation methods such as nearest-neighbor (repeating each value) or bilinear (averaging nearby pixels). The upsampled color channels are then combined with the Y channel to produce a full-color image again. Even though some color detail has been lost, the final image looks almost identical to the original to the human eye, because our visual system prioritizes brightness contrast over fine color precision.

From a storage perspective, chroma downsampling drastically reduces the amount of data that needs to be encoded and compressed. For example, with 4:2:0 sampling, the color data is reduced to just one quarter of its original size. That’s a major savings, and since the Y channel still preserves the detailed structure and edges of the image, the visual quality remains high.


In Output_part1 folder:
What each part reperesents and what is going on: Look at this table

---------------------------------+-----------------------------------+---------------------------------------------------
file                             | What’s happening	                 | Description 
---------------------------------+-----------------------------------+---------------------------------------------------
_Y.png	                         | Luminance extraction              | Brightness (fine details)  
_Cb_full.png                     | Full-resolution color channels	 | Original color difference maps    
_Cb_420view.png                  | Chroma downsampling visualization | Reduced-resolution color, then upsampled for viewing
_reconstructed_420_nearest.png   |Recombined RGB image               | Result after JPEG-style color subsampling       
---------------------------------+-----------------------------------+---------------------------------------------------

First, the image is loaded with Pillow and converted from RGB to YCbCr, which separates brightness (Y) from color information (Cb and Cr). 

NumPy is then used to handle the actual math on the color channels.

In the 4:2:2 method ('downsample_h2v1'), the code takes each pair of neighboring pixels along the width, averages them with NumPy, and keeps only half as many color samples.

In the 4:2:0 method ('downsample_h2v2'), it averages 2x2 pixel blocks, cutting both the height and width in half, leaving just one quarter of the color information. These smaller Cb and Cr arrays represent how JPEG stores less color detail to save space.

The upsampling functions then reverse this process: they use NumPy’s 'np.repeat' to duplicate each color pixel horizontally (for 4:2:2) or both horizontally and vertically (for 4:2:0), bringing the chroma maps back to the original size so they can be recombined with Y.

The reconstruction step stacks Y, Cb, and Cr again and uses Pillow to convert the data back to RGB for visualization.

Finally, the script saves several versions of the image with Pillow, the original RGB image, the reconstructed 4:2:2 and 4:2:0 versions, and the raw Y, Cb, and Cr channels, so you can see how reducing color resolution affects image quality.

Essentially, NumPy handles all the array math for downsampling and upsampling, while Pillow handles reading, color-space conversion, and saving the images.
