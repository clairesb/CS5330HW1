---
title: Mosaic_Generator
app_file: main.py
sdk: gradio
sdk_version: 5.47.2
---
# Mosaics!
## Introduction
This repository contains the code pertaining to the first homework assignment 
for CS5330 in Vancouver (Fall 2025). In order to copy this code or extend this project,
simply clone the repository and install the requirements.

## Performance Report
### Method
I wanted to go for a "roman mosaic-esque" look, so I selected tile colors automatically
based on the input image (I assume tiles would be painted or selected based on
the artist's preference a few millennia ago) and added "grout lines" between each
tile. The crux of this homework assignment was deciding when to subdivide "tiles"
into "smaller tiles"; I implemented a relatively simple policy based on color coverage
and the distance between the two most common colors in the tile.
### Metric
I chose SSIM to measure how similar tiled images were to their original counterparts,
and basically all images preformed poorly. This is not surprising, since the "grout lines"
change the visual texture of the image. I would like to implement more advanced
metrics in the future, as detailed in the next section.

## Future Work
Possible new features include making the "grout lines" optional, allowing the user
to change the subdivision policy, and creating a new similarity metric based on edge
preservation that doesn't take into account texture changes due to grouting.