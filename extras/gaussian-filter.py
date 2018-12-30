
import math
sigma = 1
kernel_size = 2*math.ceil(2*sigma)+1
channels = 1
# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
x_cord = torch.arange(kernel_size, dtype=torch.float)
x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
y_grid = x_grid.t()
xy_grid = torch.stack([x_grid, y_grid], dim=-1)

mean = (kernel_size - 1)/2.
variance = sigma**2.

# Calculate the 2-dimensional gaussian kernel which is
# the product of two gaussian distributions for two different
# variables (in this case called x and y)
gaussian_kernel = (1./(2.*math.pi*variance)) *\
    torch.exp(
              -torch.sum((xy_grid - mean)**2., dim=-1) /\
              (2*variance)
              )
# Make sure sum of values in gaussian kernel equals 1.
gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

# Reshape to 2d depthwise convolutional weight
gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=kernel_size, groups=channels, bias=False)

gaussian_filter.weight.data = gaussian_kernel
gaussian_filter.weight.requires_grad = False

res = gaussian_filter(T(P(G_y))[None])
