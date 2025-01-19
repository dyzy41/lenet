import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
import numpy as np

# 加载shp文件
shp_path = '/home/jicredt_data/dsj/CDdata/ROADCD/CRCD/change_label.shp'
gdf = gpd.read_file(shp_path)

# 定义输出栅格的分辨率
# 你可以根据需要调整这个分辨率
cell_size = 1.0

# 确定栅格图像的范围
bounds = gdf.total_bounds
x_min, y_min, x_max, y_max = bounds

# 计算栅格图像的宽度和高度
# width = int((x_max - x_min) / cell_size)
# height = int((y_max - y_min) / cell_size)
width = 32507
height = 15354

# 创建一个与目标栅格对应的transform
transform = rasterio.transform.from_bounds(x_min, y_min, x_max, y_max, width, height)

# 初始化一个栅格数组
raster = np.zeros((height, width), dtype=np.uint8)

# 将地理数据框转换为栅格
shapes = ((geom, 255) for geom in gdf.geometry)
rasterized_shapes = features.rasterize(shapes=shapes, out=raster, transform=transform)

# 保存为PNG文件
with rasterio.open(
    '/home/jicredt_data/dsj/CDdata/ROADCD/CRCD/mask.png',
    'w',
    driver='PNG',
    width=width,
    height=height,
    count=1,
    dtype=raster.dtype,
    transform=transform
) as dst:
    dst.write(rasterized_shapes, 1)

print("转换完成，生成的PNG已保存为 mask.png")
