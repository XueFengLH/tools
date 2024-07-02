# from shapely.geometry import Polygon
#
# # 定义两个多边形的点
# # polygon1_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
# # polygon2_points = [(2, 1), (3, 2), (3, 3), (2, 3)]
# polygon1_points = [(0, 0), (2, 0), (2, 2), (0, 2)]
# polygon2_points = [(1, 1), (3, 1), (3, 3), (1, 3)]
# # 创建多边形对象
# polygon1 = Polygon(polygon1_points)
# polygon2 = Polygon(polygon2_points)
#
# # 合并多边形
# merged_polygon = polygon1.union(polygon2)
# if merged_polygon.area < polygon1.area + polygon2.area:
#     merged_polygon_points = list(merged_polygon.exterior.coords)
#     print(merged_polygon_points)
#
# print("Merged Polygon:", merged_polygon)



my_list = [1, 2, 3, 4, 100464]
del my_list[4]
p = 2.5464654564561
p = int(p)
print(my_list)  # 输出 [1, 2, 4, 5]
