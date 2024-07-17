# import onnx
# from onnx.tools import update_model_dims
#
# model = onnx.load("yolov6lite_l_face.onnx")
# updated_model = update_model_dims.update_inputs_outputs_dims(model, {"inputs:0":[1,3,1920,1080]}, {"predictions:0":[1, 1025, 2049, 1]})
# onnx.save(updated_model, 'pspnet_citysc_upd.onnx')



# #
# import sys
# import argparse
# import onnx
# from onnx import shape_inference
#
#
# def conv_model_input_size(converted_model, size1, size2):
#     # print(converted_model.graph.input)
#     # exit()
#     for i, node in enumerate(converted_model.graph.input):
#         if i == 0:
#             print("Before changing input size: {}".format(node.type.tensor_type.shape))
#             print("dim:{}".format(node.type.tensor_type.shape.dim[1].dim_param))
#             if node.type.tensor_type.shape.dim[1].dim_value == 3:
#                 # NCHW
#                 node.type.tensor_type.shape.dim[2].dim_value = size1
#                 node.type.tensor_type.shape.dim[3].dim_value = size2
#             elif node.type.tensor_type.shape.dim[3].dim_value == 3:
#                 # NHWC
#                 node.type.tensor_type.shape.dim[1].dim_value = size1
#                 node.type.tensor_type.shape.dim[2].dim_value = size2
#             else:
#                 print("ERROR: Not supported input shape")
#                 return
#             print("After changing input size: {}".format(node.type.tensor_type.shape))
#         # onnx.save(converted_model, dst_fullname)
#         # exit()
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", default="./efficientnet_lite0.onnx", help="The input onnx model")
#     parser.add_argument("-a", "--action", default="size", choices=["size", "dyn"],
#                         help="Select command action")
#     parser.add_argument("-s1", "--size1", default="520", help="Input size, for action=size")
#     parser.add_argument("-s2", "--size2", default="520", help="Input size, for action=size")
#
#     args = parser.parse_args()
#     if not all([args.input, args.action]):
#         parser.print_help()
#         sys.exit(1)
#
#     src_root = args.input
#     print(src_root)
#
#     if args.action == "size":
#         if not src_root.endswith('.onnx'):
#             exit()
#
#         converted_model = onnx.load(src_root)
#         input_size1 = converted_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
#         input_size2 = converted_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
#         print("len of inputs:{}".format(len(converted_model.graph.input)))
#
#         # 修改input size
#         conv_model_input_size(converted_model, int(args.size1), int(args.size2))
#         # 疑问，函数明明没返回值，为什么converted_model模型的输入就变了呢？
#
#         # 有没有下面这一行，似乎对结果没啥影响，都只能显示输入尺寸信息，哭了！
#         inference_model = shape_inference.infer_shapes(converted_model)
#         onnx.checker.check_model(inference_model)
#         dst_fullname = src_root[:-5] + "_" + str(args.size1) + "x" + str(args.size2) + ".onnx"
#         onnx.save(inference_model, dst_fullname)
#         print(dst_fullname)
#     else:
#         print("ERROR, Invalid --action")
import onnx
# # from onnx import helper
# # from onnx import AttributeProto, TensorProto, GraphProto
#
# # 加载现有的ONNX模型
model_path = 'yolov6lite_l_face.onnx'
model = onnx.load(model_path)

# 修改输入尺寸，例如将输入的尺寸改为1x3x224x224
input_name = model.graph.input[0].name  # 假设模型的第一个输入是需要修改的输入
new_input_shape = [1, 3, 2000, 2000]

# 更新输入的形状
for i in range(len(model.graph.input[0].type.tensor_type.shape.dim)):
    model.graph.input[0].type.tensor_type.shape.dim[i].dim_value = new_input_shape[i]

# 保存修改后的模型
onnx.checker.check_model(model)  # 可以使用check_model检查模型是否有效
onnx.save(model, 'yolov6lite_l_face_1080.onnx')
