import os
import json
import shutil
path = "/mnt/sda2/x/train"
output_path = "/mnt/sda2/x/data"
img_path = output_path
for root, dirs, files in os.walk(path):
    for file in files:
        if '.jpg' in file:
            img_p = os.path.join(root, file)
            json_p = img_p.replace('.jpg', '.json')
            img_path_c = os.path.join(img_path,file)
            if os.path.exists(json_p):
                with open(json_p, "r", encoding="utf-8") as f:
                    mapping_table = json.load(f)
                    if 'shapes' in mapping_table and len(mapping_table['shapes']) > 0:
                        try:
                            # shutil.copy(img_p, img_path)
                            pass
                        except:
                            print('same file')
                    else:
                        shutil.copy(img_p, img_path_c)
            else:
                shutil.copy(img_p, img_path_c)