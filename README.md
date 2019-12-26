# GradCAM-keras
A keras implementation of Grad CAM as in <a href=https://arxiv.org/pdf/1610.02391.pdf><i> Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization </i></a>

Example of a config file:
```
{
  "model_class": "models.model.Definition",
  "loader_method": "car_color_detector",
  "weight_path": "/home/isaac/containers/ai-porsche-colour/src/model/triple_head_hist-fine-tuned-from-no-hist-1.0.2/model",
  "model_inputs": [[128,128,3], 3],
  "process": "extra.process",
  "process_inputs": ["/home/isaac/containers/Data/Car_colors/testing/fine-tuning-nouse/cognac/cognac_00004.jpg", [128,128,3], 127.5, 127.5]
}
```
