"""Data augmentation transforms for 3D MRI images.

This module provides composition-based augmentation transforms specifically
designed for 3D medical imaging, including spatial transformations, intensity
adjustments, and noise injection.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class Compose:
    """Compose multiple transforms together.
    
    Args:
        transforms: List of transform callables
    
    Example:
        >>> transform = Compose([
        ...     RandomFlip3D(p=0.5),
        ...     RandomRotation3D(degrees=15),
        ...     RandomIntensityShift(shift_range=0.1)
        ... ])
    """
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply all transforms sequentially.
        
        Args:
            image: Input tensor of shape [C, D, H, W]
        
        Returns:
            Transformed image of same shape
        """
        for transform in self.transforms:
            image = transform(image)
        return image
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomFlip3D:
    """Randomly flip 3D image along specified axes.
    
    Args:
        p: Probability of applying flip (default: 0.5)
        axes: Tuple of axes to potentially flip (default: (1, 2, 3) for D, H, W)
    
    Example:
        >>> transform = RandomFlip3D(p=0.5, axes=(2, 3))  # Only flip H and W
        >>> flipped = transform(image)
    """
    
    def __init__(self, p: float = 0.5, axes: Tuple[int, ...] = (1, 2, 3)):
        self.p = p
        self.axes = axes
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random flips.
        
        Args:
            image: Input tensor [C, D, H, W]
        
        Returns:
            Flipped image [C, D, H, W]
        """
        for axis in self.axes:
            if torch.rand(1).item() < self.p:
                image = torch.flip(image, dims=[axis])
        return image
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p}, axes={self.axes})'


class RandomRotation3D:
    """Randomly rotate 3D image around specified axes.
    
    Args:
        degrees: Range of rotation angles in degrees
            - If float: rotate in [-degrees, degrees]
            - If tuple: rotate in [degrees[0], degrees[1]]
        axes_plane: Plane to rotate in, either 'xy', 'xz', or 'yz' (default: 'xy')
        p: Probability of applying rotation (default: 0.5)
    
    Example:
        >>> transform = RandomRotation3D(degrees=15, axes_plane='xy', p=0.5)
    """
    
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        axes_plane: str = 'xy',
        p: float = 0.5,
    ):
        self.p = p
        self.axes_plane = axes_plane
        
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random rotation.
        
        Args:
            image: Input tensor [C, D, H, W]
        
        Returns:
            Rotated image [C, D, H, W]
        """
        if torch.rand(1).item() >= self.p:
            return image
        
        # Sample rotation angle
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        angle_rad = np.deg2rad(angle)
        
        # Create rotation matrix
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        
        # Build affine matrix for 2D rotation in specified plane
        theta = torch.tensor([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0]
        ], dtype=image.dtype, device=image.device).unsqueeze(0)
        
        # For 3D image, we rotate each 2D slice
        if self.axes_plane == 'xy':
            # Rotate in H-W plane (last two dimensions)
            C, D, H, W = image.shape
            rotated = []
            for d in range(D):
                slice_2d = image[:, d:d+1, :, :]  # [C, 1, H, W]
                grid = F.affine_grid(theta, slice_2d.size(), align_corners=False)
                rotated_slice = F.grid_sample(
                    slice_2d, grid, mode='bilinear', padding_mode='border',
                    align_corners=False
                )
                rotated.append(rotated_slice)
            image = torch.cat(rotated, dim=1)
        
        return image
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(degrees={self.degrees}, '
                f'axes_plane={self.axes_plane}, p={self.p})')


class RandomIntensityShift:
    """Randomly shift image intensity values.
    
    Applies additive shift: I' = I + shift
    
    Args:
        shift_range: Range of intensity shift
            - If float: shift in [-shift_range, shift_range]
            - If tuple: shift in [shift_range[0], shift_range[1]]
        p: Probability of applying shift (default: 0.5)
    
    Example:
        >>> transform = RandomIntensityShift(shift_range=0.1, p=0.5)
    """
    
    def __init__(
        self,
        shift_range: Union[float, Tuple[float, float]],
        p: float = 0.5,
    ):
        self.p = p
        
        if isinstance(shift_range, (int, float)):
            self.shift_range = (-shift_range, shift_range)
        else:
            self.shift_range = shift_range
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random intensity shift.
        
        Args:
            image: Input tensor [C, D, H, W]
        
        Returns:
            Shifted image [C, D, H, W]
        """
        if torch.rand(1).item() >= self.p:
            return image
        
        shift = np.random.uniform(self.shift_range[0], self.shift_range[1])
        return torch.clamp(image + shift, 0, 1)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(shift_range={self.shift_range}, p={self.p})'


class RandomIntensityScale:
    """Randomly scale image intensity values.
    
    Applies multiplicative scaling: I' = I * scale
    
    Args:
        scale_range: Range of intensity scaling factors
            - If tuple: scale in [scale_range[0], scale_range[1]]
        p: Probability of applying scaling (default: 0.5)
    
    Example:
        >>> transform = RandomIntensityScale(scale_range=(0.9, 1.1), p=0.5)
    """
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,
    ):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random intensity scaling.
        
        Args:
            image: Input tensor [C, D, H, W]
        
        Returns:
            Scaled image [C, D, H, W]
        """
        if torch.rand(1).item() >= self.p:
            return image
        
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return torch.clamp(image * scale, 0, 1)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(scale_range={self.scale_range}, p={self.p})'


class RandomGaussianNoise:
    """Add random Gaussian noise to image.
    
    Args:
        noise_std: Standard deviation of Gaussian noise
            - If float: std = noise_std
            - If tuple: std sampled from [noise_std[0], noise_std[1]]
        p: Probability of adding noise (default: 0.5)
    
    Example:
        >>> transform = RandomGaussianNoise(noise_std=0.01, p=0.3)
    """
    
    def __init__(
        self,
        noise_std: Union[float, Tuple[float, float]] = 0.01,
        p: float = 0.5,
    ):
        self.p = p
        
        if isinstance(noise_std, (int, float)):
            self.noise_std = (noise_std, noise_std)
        else:
            self.noise_std = noise_std
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to image.
        
        Args:
            image: Input tensor [C, D, H, W]
        
        Returns:
            Noisy image [C, D, H, W]
        """
        if torch.rand(1).item() >= self.p:
            return image
        
        std = np.random.uniform(self.noise_std[0], self.noise_std[1])
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0, 1)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(noise_std={self.noise_std}, p={self.p})'


class RandomGaussianBlur:
    """Apply random Gaussian blur to image.
    
    Args:
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation range for Gaussian kernel
            - If float: sigma = sigma
            - If tuple: sigma sampled from [sigma[0], sigma[1]]
        p: Probability of applying blur (default: 0.5)
    
    Example:
        >>> transform = RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.3)
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        sigma: Union[float, Tuple[float, float]] = (0.1, 2.0),
        p: float = 0.5,
    ):
        self.kernel_size = kernel_size
        self.p = p
        
        if isinstance(sigma, (int, float)):
            self.sigma = (sigma, sigma)
        else:
            self.sigma = sigma
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur.
        
        Args:
            image: Input tensor [C, D, H, W]
        
        Returns:
            Blurred image [C, D, H, W]
        """
        if torch.rand(1).item() >= self.p:
            return image
        
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        
        # Create 2D Gaussian kernel
        kernel_1d = self._get_gaussian_kernel_1d(self.kernel_size, sigma)
        kernel_2d = kernel_1d.unsqueeze(-1) * kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        
        # Apply blur to each 2D slice
        C, D, H, W = image.shape
        blurred = []
        for d in range(D):
            slice_2d = image[:, d:d+1, :, :]  # [C, 1, H, W]
            # Expand kernel for all channels
            kernel = kernel_2d.repeat(C, 1, 1, 1).to(image.device)
            blurred_slice = F.conv2d(
                slice_2d.unsqueeze(0),
                kernel,
                padding=self.kernel_size // 2,
                groups=C
            ).squeeze(0)
            blurred.append(blurred_slice)
        
        return torch.cat(blurred, dim=1)
    
    def _get_gaussian_kernel_1d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel.
        
        Args:
            kernel_size: Kernel size (odd number)
            sigma: Standard deviation
        
        Returns:
            1D Gaussian kernel [kernel_size]
        """
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - (kernel_size - 1) / 2
        kernel = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        return kernel / kernel.sum()
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(kernel_size={self.kernel_size}, '
                f'sigma={self.sigma}, p={self.p})')


def get_train_transforms(
    flip_prob: float = 0.5,
    rotation_degrees: float = 15.0,
    intensity_shift: float = 0.1,
    intensity_scale: Tuple[float, float] = (0.9, 1.1),
    noise_std: float = 0.01,
    blur_prob: float = 0.3,
) -> Compose:
    """Get standard training augmentation pipeline.
    
    Args:
        flip_prob: Probability for random flips
        rotation_degrees: Max rotation angle in degrees
        intensity_shift: Max intensity shift range
        intensity_scale: Intensity scaling factor range
        noise_std: Gaussian noise standard deviation
        blur_prob: Probability for Gaussian blur
    
    Returns:
        Composed augmentation transforms
    
    Example:
        >>> train_transform = get_train_transforms(
        ...     flip_prob=0.5,
        ...     rotation_degrees=10,
        ...     intensity_shift=0.1
        ... )
    """
    return Compose([
        RandomFlip3D(p=flip_prob, axes=(2, 3)),  # Flip H and W
        RandomRotation3D(degrees=rotation_degrees, axes_plane='xy', p=0.5),
        RandomIntensityShift(shift_range=intensity_shift, p=0.5),
        RandomIntensityScale(scale_range=intensity_scale, p=0.5),
        RandomGaussianNoise(noise_std=noise_std, p=0.3),
        RandomGaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=blur_prob),
    ])


def get_val_transforms() -> Optional[Compose]:
    """Get validation/test augmentation pipeline (typically none).
    
    Returns:
        None (no augmentation for validation/test)
    
    Example:
        >>> val_transform = get_val_transforms()  # Returns None
    """
    return None
