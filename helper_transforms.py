import torch 
import torch.nn.functional as F
import torchvision.transforms as transforms

### FOR GENERATORS
# Illumination consistency
class IlluminationTransform(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, image):
        # Ensure the image is in the range [0, 1]
        image = torch.clamp(image, 0, 1)

        # Apply gamma transformation
        image = image ** self.gamma

        return image


# For geometric consistency
class GeometricTransform(object):
    def __init__(self, k=1):
        self.k = k

    def __call__(self, image):
        # Rotate tensor by 90 degrees k times
        rotated_image = torch.rot90(image, self.k, (1, 2))

        return rotated_image
    
    
class InverseGeometricTransform(object):
    def __init__(self, k=1):
        self.k = k

    def __call__(self, rotated_image):
        # Rotate tensor by -90 degrees k times to invert the rotation
        original_image = torch.rot90(rotated_image, -self.k, (1, 2))

        return original_image
    
    
### FOR DISCRIMINATORS
# 1. Color iamge using blurring
def gaussian_blur_tensor(image, kernel_size=3, sigma=1.0):
    # Convert image to float tensor
    image_tensor = image.float()

    # Define Gaussian kernel
    kernel = torch.tensor([[torch.exp(-((i - kernel_size // 2) ** 2 + (j - kernel_size // 2) ** 2) / (2 * sigma ** 2))
                            for j in range(kernel_size)] for i in range(kernel_size)])

    # Normalize kernel
    kernel = kernel / kernel.sum()

    # Add batch and channel dimensions
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Apply convolution with Gaussian kernel
    blurred_image = F.conv2d(image_tensor.unsqueeze(0), kernel, padding=kernel_size // 2)

    return blurred_image.squeeze(0)

# 2. Texture image using a grayscale image
def get_grayscale_tensor(image):
    """
    Convert an image to grayscale.

    Args:
        image: PIL image or PyTorch tensor.

    Returns:
        Grayscale image as a PyTorch tensor.
    """
    grayscale_transform = transforms.Grayscale()
    
    if isinstance(image, torch.Tensor):
        # If input is a tensor, convert it to PIL image first
        image_pil = transforms.ToPILImage()(image)
        grayscale_image = grayscale_transform(image_pil)
        grayscale_tensor = transforms.ToTensor()(grayscale_image)
    else:
        grayscale_image = grayscale_transform(image)
        grayscale_tensor = transforms.ToTensor()(grayscale_image)
        
    return grayscale_tensor


# 3. Edge image using a Prewit Operator
def prewitt_edge_detection_tensor(image):
    # Define Prewitt kernels
    kernel_x = torch.FloatTensor([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)

    kernel_y = torch.FloatTensor([[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]]).unsqueeze(0).unsqueeze(0)

    # Convert image to float tensor and add batch dimension
    image_tensor = image.float().unsqueeze(0).unsqueeze(0)

    # Apply convolution with Prewitt kernels
    edge_x = F.conv2d(image_tensor, kernel_x, padding=1)
    edge_y = F.conv2d(image_tensor, kernel_y, padding=1)

    # Compute magnitude of gradients
    edge = torch.sqrt(edge_x**2 + edge_y**2)

    # Normalize values to 0-1 range
    edge = edge / edge.max()

    return edge.squeeze(0).squeeze(0)
