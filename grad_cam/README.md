## Grad-CAM
- Original Impl: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- Grad-CAM: [https://b23.tv/1kccjmb](https://b23.tv/1kccjmb)
- How to use the Grad-CAM: [https://b23.tv/n1e60vN](https://b23.tv/n1e60vN)

## Use a process (replace it with your own network)
1. Replace the code that created the model with the code that created the model itself, and load your own trained weights
2. Set the appropriate `target_layers` for your network
3. Set up appropriate preprocessing methods according to your network
4. Assign the predicted image path to `img_path`
5. Assign the category id of interest to `target_category`

