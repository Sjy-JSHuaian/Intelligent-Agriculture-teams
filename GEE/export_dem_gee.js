// Google Earth Engine：导出元阳研究区 Copernicus DEM GLO-30
// 使用前提：拥有 Google Earth Engine 账号并在 https://code.earthengine.google.com/ 登录授权。
// 说明：GEE 的 DEM 目录需要账号授权，不能像 AWS 镜像那样直接匿名下载；
//       本项目默认使用 AWS Open Data 的同源 GLO-30 瓦片，GEE 仅作为备选/对照。

var aoi = ee.Geometry.Rectangle([102.690, 23.030, 102.782, 23.132]);  // west, south, east, north

var dem = ee.Image('COPERNICUS/DEM/GLO30').select('DEM').clip(aoi);

Map.centerObject(aoi, 12);
Map.addLayer(dem, {min: 700, max: 2200, palette: ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}, 'GLO30 DEM');

Export.image.toDrive({
  image: dem,
  description: 'yuanyang_GLO30_30m',
  folder: 'dachuang_memberA',
  fileNamePrefix: 'yuanyang_GLO30_30m',
  region: aoi,
  scale: 30,
  crs: 'EPSG:32648',
  maxPixels: 1e9
});

// 导出完成后：Google Drive → dachuang_memberA → 下载 .tif
// 也可用 Export.image.toAsset 存为 GEE Asset。
