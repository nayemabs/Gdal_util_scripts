import rasterio
import numpy as np
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.coords import BoundingBox

# Paths to input and output raster files
dhaka_input_path = "dhaka_ground_truth.tif"
wb_path = "WB_padded.tif"
dhaka_output_path = "dhaka_ground_truth_masked.tif"
wb_output_path = "WB_masked.tif"

# Open both rasters
with rasterio.open(dhaka_input_path) as dhaka_src, rasterio.open(wb_path) as wb_src:
    # Get bounds (left, bottom, right, top)
    dhaka_bounds = dhaka_src.bounds
    wb_bounds = wb_src.bounds

    # Find intersection bounding box
    intersection_bounds = BoundingBox(
        max(dhaka_bounds.left, wb_bounds.left),
        max(dhaka_bounds.bottom, wb_bounds.bottom),
        min(dhaka_bounds.right, wb_bounds.right),
        min(dhaka_bounds.top, wb_bounds.top)
    )

    # Crop both images to the intersection area
    dhaka_window = from_bounds(*intersection_bounds, transform=dhaka_src.transform)
    dhaka_data = dhaka_src.read(window=dhaka_window)
    dhaka_transform = dhaka_src.window_transform(dhaka_window)

    wb_window = from_bounds(*intersection_bounds, transform=wb_src.transform)
    wb_data = wb_src.read(window=wb_window)
    wb_transform = wb_src.window_transform(wb_window)

    # Ensure both rasters have the same shape
    dhaka_height, dhaka_width = dhaka_data.shape[1], dhaka_data.shape[2]
    wb_height, wb_width = wb_data.shape[1], wb_data.shape[2]

    # If WB does not match Dhaka shape, resize WB
    if (wb_height, wb_width) != (dhaka_height, dhaka_width):
        wb_resized = np.zeros_like(dhaka_data, dtype=wb_data.dtype)
        reproject(
            wb_data,
            wb_resized,
            src_transform=wb_transform,
            dst_transform=dhaka_transform,
            src_crs=wb_src.crs,
            dst_crs=dhaka_src.crs,
            resampling=Resampling.nearest
        )
    else:
        wb_resized = wb_data

    # Create a binary mask (1 where WB has valid data, 0 otherwise)
    wb_mask = (wb_resized > 0).astype(dhaka_data.dtype)

    # Apply mask: Keep intersection, set non-overlapping areas to 0
    dhaka_masked = dhaka_data * wb_mask
    wb_masked = wb_resized * wb_mask  # Ensures WB also has 0 in non-overlapping areas

    # Save masked Dhaka raster
    with rasterio.open(
        dhaka_output_path, "w", driver="GTiff",
        height=dhaka_height, width=dhaka_width,
        count=dhaka_src.count, dtype=dhaka_data.dtype,
        crs=dhaka_src.crs, transform=dhaka_transform
    ) as dst:
        dst.write(dhaka_masked)

    # Save masked WB raster
    with rasterio.open(
        wb_output_path, "w", driver="GTiff",
        height=dhaka_height, width=dhaka_width,
        count=wb_src.count, dtype=wb_data.dtype,
        crs=wb_src.crs, transform=dhaka_transform
    ) as dst:
        dst.write(wb_masked)

print("Cropping and masking complete!")
