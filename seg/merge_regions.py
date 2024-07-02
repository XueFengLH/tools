from shapely.geometry import Polygon
import json
import os
from tqdm import tqdm
def qz()
rounded_list = [round(num) for num in original_list]
def r_json(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    shapes = data['shapes']
    n = len(shapes)
    if n > 1:
        print(file_path)
        print(data)
        # print(data)
        b = 1
        while b > 0:
            n = len(shapes)
            if n == 1:
                break
            b = 0
            for i,shape1 in enumerate(shapes):
                for j,shape2 in enumerate(shapes[i+1:]):
                    polygon1_points = [tuple(int(point)) for point in shape1['points']]
                    polygon2_points = [tuple(int(point)) for point in shape2['points']]
                    polygon1 = Polygon(polygon1_points)
                    polygon2 = Polygon(polygon2_points)
                    merged_polygon = polygon1.union(polygon2)
                    if merged_polygon.area < polygon1.area + polygon2.area:
                        try:
                            merged_polygon_points = list(merged_polygon.exterior.coords)
                        except:
                            continue
                        shapes[i]['points'] = merged_polygon_points
                        del shapes[i+j+1]
                        b = 1
                        print(1)
        data['shapes'] = shapes
        file.close()
        with open(file_path , 'w', encoding='utf-8') as file1:
                json.dump(data, file1, indent=4)



if __name__ == '__main__':
    path = '/mnt/sda2/test_seg/json_labels'
    dir_list = os.listdir(path)
    for dir in tqdm(dir_list):
        file_path = os.path.join(path, dir)
        r_json(file_path)


