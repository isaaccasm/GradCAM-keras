# GradCAM-keras
A keras implementation of Grad CAM as in <a href=https://arxiv.org/pdf/1610.02391.pdf><i> Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization </i></a>

It includes a new addition that is a separation between the positive and negative part of the image. This means that that two images will be shown one with part that represents the part that make the model selects an specific class and the parts that made the model believes it was a different class. 

The code is created to be called using command line. The help option can be used by calling:
`python gradcam.py -help`

##Bash call
A typical call is:
```
python gradcam.py -model_args_path /path/to/file.json --guided -layer_name conv2d -last_layer out
```

##Standard input files
Example of a config file (file used in `-model_args_path`):
```
{
  "model_class": "models.model_car_color",
  "loader_method": "car_color_detector",
  "weight_path": "path/to/keras/weights/file",
  "model_inputs": [[128,128,3], 3],
  "process": "models.process",
  "process_inputs": ["/path/to/image/to/test", [128,128,3], 127.5, 127.5]
}
```
The process input may be different and it does not have any standard.


