#!/usr/bin/env python3
import os
import sys
import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
import yaml
import cv2
import glob
import warnings
warnings.filterwarnings('ignore')

class YOLOv9ToTensorRT:
    def __init__(self, model_path, data_yaml, imgsz=640, batch=1, device='cuda:0'):
        self.model_path = Path(model_path)
        self.data_yaml = Path(data_yaml)
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.workspace_size = 8 << 30  # 8GB
        with open(self.data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        self.num_classes = self.data_config.get('nc', 80)
        self.logger = trt.Logger(trt.Logger.INFO)

    def export_to_onnx(self):
        from ultralytics import YOLO
        model = YOLO(str(self.model_path))
        pt_model = model.model
        pt_model.eval().float().to(self.device)

        dummy_input = torch.randn(self.batch, 3, self.imgsz, self.imgsz).to(self.device)
        onnx_path = self.model_path.with_suffix('.onnx')

        print(f"Exporting ONNX to {onnx_path}")
        torch.onnx.export(
            pt_model,
            dummy_input,
            str(onnx_path),
            opset_version=12,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )
        return onnx_path

    def build_engine(self, onnx_path, precision='fp16'):
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)

        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)

        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        profile = builder.create_optimization_profile()
        profile.set_shape("images",
                          (1, 3, self.imgsz, self.imgsz),
                          (self.batch, 3, self.imgsz, self.imgsz),
                          (8, 3, self.imgsz, self.imgsz))
        config.add_optimization_profile(profile)

        print("Building TensorRT engine... (640px input)")
        engine_path = self.model_path.with_suffix('.engine')
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine:
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            print(f"✅ Saved engine to: {engine_path}")
            return engine_path
        else:
            print("❌ Engine build failed.")
            return None

    def validate_engine(self, engine_path):
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if not engine:
            print("❌ Engine validation failed.")
            return False
        context = engine.create_execution_context()
        if not context:
            print("❌ Execution context creation failed.")
            return False
        print("✅ Engine validated successfully.")
        return True

    def convert(self, precision='fp16'):
        print("⚙️ Starting conversion...")
        onnx_path = self.export_to_onnx()
        engine_path = self.build_engine(onnx_path, precision)
        if engine_path and self.validate_engine(engine_path):
            print(f"🎉 Conversion to TensorRT {precision} complete.")
            return engine_path
        return None


def main():
    model_path = "models/128060ep.pt"
    data_yaml = "data/football-players-v13/data.yaml"
    imgsz = 640
    batch = 1

    converter = YOLOv9ToTensorRT(model_path, data_yaml, imgsz, batch)

    for precision in ['fp16', 'fp32']:
        print(f"\n== Attempting {precision} ==")
        if converter.convert(precision=precision):
            break
    else:
        print("❌ All conversion attempts failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
