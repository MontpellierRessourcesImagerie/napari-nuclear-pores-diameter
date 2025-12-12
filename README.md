# Napari: Nuclear pores diameter

## Images

- Images must be TIFF image.
- Only 2D is handled for now.

## Use the plugin

- Open Napari, launch the plugin and open one of your images.

### Pattern generation

- `Radius`: Radius of a theoretical nuclear pore.
- `Anisotropy`: Anisotropy ratio of a pore (1.0 = circle, <1.0 = bigger on Y axis, >1.0 = bigger on X axis).
- `Padding`: Number of padding pixels around the kernel to allow the blur to propagate.
- `Sigma`: Sigma of the Gaussian used for Gaussian blur.
- Click on "Make pattern" to generate the theoretical pattern, it should show up in the viewer.

### Nuclear pores detection

- `Method`: Method used to perform pattern recognition of the image.
- `Prominence`: Prominence of peaks on the convoluted image for this location to be considered as a center.
- Click of the image layer and click "Detect nuclear pores" to launch detection. Centers should show up as a points layer.

### Analyze

- `Pixel Size (µm)`: Physical size of a pixel in µm.
- `Initial diameter (µm)`: Maximal diameter of a nuclear pore +10%.
- `Number of steps`: How many radial profiles are computed to make the average profile.
- `Method`: Method used to process the diameter of a pore (by peaks or gaussian fitting).
- `Export plots`: Should we export one image per pore showing the intensity profile and the peaks.
- `Select working directory`: Allows you to choose an empty directory into which measures and plots will be exported.
- You can now click on the image and then on "Run radial analysis". Napari should freeze during a minute or two. You should see the folder getting filled with plots.

