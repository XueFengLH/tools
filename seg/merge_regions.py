from shapely.geometry import Polygon
import json
import os
from tqdm import tqdm
def int_list(l):
    rounded_list = [round(num) for num in l]
    return rounded_list
def r_json(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    shapes = data['shapes']
    n = len(shapes)
    if n > 1:
        # print(file_path)
        # print(data)
        # print(data)
        b = 1
        while b > 0:
            n = len(shapes)
            if n == 1:
                break
            b1 = 0
            b = 0
            for i in range(n):
                if i >= n - b1:
                    break
                for j in range(i+1,n):
                    if i + j >= n - b1:
                        break
                    shape1 = shapes[i]
                    shape2 = shapes[j]
                    polygon1_points = [tuple(int_list(point)) for point in shape1['points']]
                    polygon2_points = [tuple(int_list(point)) for point in shape2['points']]
                    polygon1 = Polygon(polygon1_points)
                    polygon2 = Polygon(polygon2_points)
                    try:
                        merged_polygon = polygon1.union(polygon2)
                    except:
                        continue
                    if merged_polygon.area < polygon1.area + polygon2.area:
                        try:
                            merged_polygon_points = list(merged_polygon.exterior.coords)
                        except:
                            continue
                        shapes[i]['points'] = merged_polygon_points
                        del shapes[i+j]
                        b1 = b1 + 1
                        # print(i+j)
                        b = 1
        data['shapes'] = shapes
        file.close()
        with open(file_path , 'w', encoding='utf-8') as file1:
                json.dump(data, file1, indent=4)



if __name__ == '__main__':
    path = '/mnt/sda2/code/ultralytics/data_2/json_labels'
    dir_list = os.listdir(path)
    for dir in tqdm(dir_list):
        file_path = os.path.join(path, dir)
        r_json(file_path)


