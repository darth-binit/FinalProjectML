import torch
import torch.nn as nn
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model (nn.Module): Your trained model.
            target_layer (nn.Module): The layer (module) in your model where you want to visualize.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        if self.target_layer is not None:
            self._register_hooks()

    def set_target_layer(self, target_layer):
        # Optionally remove old hooks if you want to switch layers
        self.target_layer = target_layer
        self._register_hooks()

    def _save_activations_hook(self, module, input, output):
        self.activations = output.clone().detach()

    def _save_gradients_hook(self, module, grad_in, grad_out):
        # grad_out is a tuple; we want grad_out[0]
        self.gradients = grad_out[0].clone().detach()

    def _register_hooks(self):
        # Make sure self.target_layer is not None
        if self.target_layer is not None:
            self.target_layer.register_forward_hook(self._save_activations_hook)
            self.target_layer.register_backward_hook(self._save_gradients_hook)
        else:
            raise ValueError("target_layer is None; cannot register hooks.")

    def generate_cam(self, input_tensor, class_idx=None):
        """
        Args:
            input_tensor (torch.Tensor): A batch of images, shape (B, C, H, W).
            class_idx (int, optional): The class index for which we compute Grad-CAM.
                                       If None, we use the predicted class of the first sample.
        Returns:
            cam (numpy.ndarray): The heatmap (H, W) in [0,1] range for the first image in the batch.
        """
        # Forward pass
        output = self.model(input_tensor)  # shape (B, num_classes)

        if class_idx is None:
            # Use the predicted class of the first sample
            class_idx = output.argmax(dim=1)[0].item()

        # Zero existing gradients
        self.model.zero_grad()

        # Backward pass for the chosen class
        # We pick the first sample in the batch for demonstration
        target = output[0, class_idx]
        target.backward()

        # Get the gradients and activations we saved
        gradients = self.gradients[0]  # shape (num_filters, H', W')
        activations = self.activations[0]  # shape (num_filters, H', W')

        # Compute the mean of the gradients for each filter
        alpha = gradients.mean(dim=(1, 2), keepdim=True)  # shape (num_filters, 1, 1)

        # Weighted sum of activations
        weighted_activations = alpha * activations
        cam = weighted_activations.sum(dim=0)  # shape (H', W')

        # ReLU
        cam = torch.clamp(cam, min=0)

        # Normalize to [0,1] for visualization
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam_np = cam.detach().cpu().numpy()

        return cam_np  # shape (H', W'), in [0,1]

def compute_cam_and_overlay(model, target_layer, input_tensor, pil_img, gradcam, class_idx=None):
    """
    Run Grad-CAM for a given target layer, return (heatmap, overlayed) as NumPy arrays in RGB.
    """
    # 1. Update the GradCAM instance to hook the chosen layer
    gradcam.target_layer = target_layer

    # 2. Generate CAM
    cam_np = gradcam.generate_cam(input_tensor, class_idx=class_idx)  # shape: (H', W') in [0,1]

    # 3. Create color heatmap
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 4. Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap_rgb, (pil_img.width, pil_img.height))

    # 5. Overlay
    original_np = np.array(pil_img)  # shape (H, W, 3) in RGB
    overlayed = cv2.addWeighted(original_np, 0.5, heatmap_resized, 0.5, 0)

    return heatmap_resized, overlayed


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.0, weight=None):
        """
        FocalLoss that combines label smoothing and class weighting.

        Args:
            alpha (float): Scaling factor.
            gamma (float): Focusing parameter.
            label_smoothing (float): Amount of label smoothing.
            weight (Tensor): A manual rescaling weight given to each class.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(reduction='none',
                                      label_smoothing=label_smoothing,
                                      weight=weight)

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)  # probability for the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()