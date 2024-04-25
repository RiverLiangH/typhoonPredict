import folium

# 给定的路径数据
path_data = [
    [131.7, 7.8],
    [135.9, 5.5],
    [133.7, 9.1],
    [136.1, 14.8],
    [138.9, 12.4],
    [134.4, 10.2],
    [133.4, 8.9],
    [135.2, 10.8],
    [136.6, 5.6],
    [135.6, 15.3],
    [136.2, 5.5],
    [139.6, 12.6],
    [134.2, 9.9]
]

# 模型预测的路径数据（示例数据）
predicted_path_data = [
    [-5.314434,   -0.5568044],
    [-7.245735,   -3.1701784],
    [-7.1286983,  -5.3798738],
    [-4.781143,   -3.722022],
    [-5.630187,   -3.922287],
    [-6.816636,   -4.9502177],
    [-6.0122356,  -5.600479],
    [-7.2681684,  -4.7853794],
    [-5.190033,   -3.5009117],
    [-7.242642,   -3.3939607],
    [-5.936207,   -4.0683317],
    [-7.238674,   -5.073143],
    [-5.6687627,  -5.5592775]
]
longitudes = [point[0] for point in path_data]
latitudes = [point[1] for point in path_data]
# 创建地图对象
m = folium.Map(location=[(max(longitudes) + min(longitudes)) / 2, (max(latitudes) + min(latitudes)) / 2], zoom_start=5)

# 添加原始路径
folium.PolyLine(path_data, color="blue", weight=2.5, opacity=1).add_to(m)

# 添加模型预测的路径
folium.PolyLine(predicted_path_data, color="red", weight=2.5, opacity=1).add_to(m)

# 保存地图为html文件
m.save("path_map.html")
