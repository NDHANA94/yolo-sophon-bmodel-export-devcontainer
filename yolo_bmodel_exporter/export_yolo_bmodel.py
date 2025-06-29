#!/usr/bin/env python3

# @author: WM Nipun Dhananjaya Weerakkodi
# @email: nipun.dhananjaya@gmail.com
# @date: 2025-07-29

import argparse
import os
import sys
import re
import types
import shutil
import onnx
from onnx import ModelProto
from ultralytics import YOLO
from tpu_mlir.entry import model_transform
from tpu_mlir.entry import model_deploy
from tpu_mlir.entry import run_calibration
import yaml
import subprocess

# Custom logging functions to print messages 
def log_info(message):
    print(f"\033[92m{message}\033[0m")  # Green text for info messages

def log_error(message):
    print(f"\033[91m{message}\033[0m")  # Red text for error messages



class YOLOBModelExporter:
    def __init__(self, config_path: str):
        self.model_name: str = None                 # will be updated after onnx export
        self.model_name: str = None                 # will be updated after onnx export
        self.model_export_dir: str = "models"       # Directory to save exported models
        self.mlir_export_dir: str = "mlir"          # Directory to save mlir files
        self.bmodel_export_dir: str = "bmodel"      # Directory to save bmodel files
        # parameters for mlir export
        self.model_input_shape: tuple = None        # will be updated after onnx export
        self.model_output_names: list = None        # will be updated after onnx export
        self.img_ch_mean: list = [0.0, 0.0, 0.0]    # Default mean for RGB images
        self.img_ch_scale: list = [1.0/255.0]*3     # Default scale for RGB images
        self.pixel_format: str = "rgb"              # Default pixel format
        self.keep_aspect_ratio: bool = True         # Default to keep aspect ratio
        self.mlir_test_input: str = None            # Path to the test input image
        # parameters for bmodel export
        self.bmodel_quantize: str = None            # Quantization method;  F32/BF16/F16/INT8
        self.bmodel_processor: str = None           # Processor type; bm1684x/bm1684/bm1688/cv186x/bm1690
        self.bmodel_tolerance: list = None          # Tolerance for nmodel export
        # calibration parameters
        self.calib_dataset: str = None              # Path to the calibration dataset
        self.calib_input_num: int = 100             # Number of images to use for calibration
        self.calib_output: str = None               # Path to save calibration output

        # exported model paths
        self.pt_model_path: str = None              # Get updated after calling self.__move_models_to_export_dir()
        self.onnx_model_path: str = None            # Get updated after calling self.__move_models_to_export_dir()
        self.mlir_model_path: str = None            # Get updated after calling self.export_mlir()
        self.bmodel_path: str = None                # Get updated after calling self.export_bmodel()

        # Load the config, pt model, and export to ONNX
        self.__load_config(config_path)
        self.pt_model = YOLO(self.model_name+".pt")
        self.onnx_model: ModelProto = self.__export_onnx()
        self.__create_export_dir()
        self.__move_models_to_export_dir()


    def __load_config(self, config_path: str):
        log_info("===================================================================")
        log_info(">> Loading configuration from YAML file...")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                # model name and export directory
                self.model_name = config.get("model_name", None)
                if not self.model_name:
                    raise ValueError("`model_name` is required in the configuration file.")
                # parameters for mlir export
                self.img_ch_mean = config.get("mlir_export_config", {}).get("img_ch_mean", self.img_ch_mean)
                self.img_ch_scale = config.get("mlir_export_config", {}).get("img_ch_scale", self.img_ch_scale)
                self.keep_aspect_ratio = config.get("mlir_export_config", {}).get("keep_aspect_ratio", self.keep_aspect_ratio)
                self.pixel_format = config.get("mlir_export_config", {}).get("pixel_format", self.pixel_format)
                self.mlir_test_input = config.get("mlir_export_config", {}).get("test_input", self.mlir_test_input)
                if self.img_ch_mean is None or not isinstance(self.img_ch_mean, list) or len(self.img_ch_mean) != 3:
                    raise ValueError("Invalid img_ch_mean configuration. It should be a list of three float values.")
                if self.img_ch_scale is None or not isinstance(self.img_ch_scale, list) or len(self.img_ch_scale) != 3:
                    raise ValueError("Invalid img_ch_scale configuration. It should be a list of three float values.")
                if self.mlir_test_input and not os.path.exists(self.mlir_test_input):
                    raise ValueError(f"Test input file {self.test_input} does not exist.")
                if self.pixel_format not in ["rgb", "bgr"]:
                    raise ValueError("Invalid pixel format. Supported formats are 'rgb' and 'bgr'.")
                if self.keep_aspect_ratio not in [True, False]:
                    raise ValueError("keep_aspect_ratio should be a boolean value (true or false).")
                # parameters for bmodel export
                self.bmodel_quantize = config.get("bmodel_export_config", {}).get("quantize", self.bmodel_quantize)
                self.bmodel_processor = config.get("bmodel_export_config", {}).get("processor", self.bmodel_processor)
                self.bmodel_tolerance = config.get("bmodel_export_config", {}).get("tolerance", self.bmodel_tolerance)
                if self.bmodel_tolerance[0] < 0 and self.bmodel_tolerance[1] < 0:
                    self.bmodel_tolerance = None  # If both values are negative, set to None
                self.calib_dataset = config.get("calibration_config", {}).get("dataset", self.calib_dataset)
                self.calib_input_num = config.get("calibration_config", {}).get("input_num", self.calib_input_num)
                if self.bmodel_quantize not in ["F32", "BF16", "F16", "INT8"]:
                    raise ValueError("Invalid quantization method. Supported methods are 'F32', 'BF16', 'F16', and 'INT8'.")
                if self.bmodel_processor not in ["bm1684x", "bm1684", "bm1688", "cv186x", "bm1690"]:
                    raise ValueError("Invalid processor type. Supported types are 'bm1684x', 'bm1684', 'bm1688', 'cv186x', and 'bm1690'.")
                if self.bmodel_quantize == "INT8" and self.calib_dataset and not os.path.exists(self.calib_dataset):
                    raise ValueError(f"Calibration dataset {self.calib_dataset} does not exist.")
                if not isinstance(self.calib_input_num, int) or self.calib_input_num <= 0:
                    raise ValueError("calib_input_num should be a positive integer.")
        except Exception as e:
            log_error(f">> Error loading configuration file {config_path}: {e}")
            sys.exit(1)
        log_info(">> Configuration loaded successfully.")
        log_info("-------------------------------------------------------------------\n")
        

    def __export_onnx(self):
        log_info("\n==================================================================")
        log_info(">> Exporting model to ONNX format...")
        if self.pt_model is None:
            log_error(">> Model not loaded. Please load the model first using load_pt_model().")
            sys.exit(1)
        # export to ONNX format
        result = self.pt_model.export(format="onnx")
        if not result:
            log_error(">> Failed to export model to ONNX format.")
            sys.exit(1)
        # load exported onnx model and get input shape and output names
        self.onnx_model = onnx.load(result)
        self.model_input_shape = self.onnx_model.graph.input[0].type.tensor_type.shape.dim
        self.model_input_shape = tuple(dim.dim_value for dim in self.model_input_shape)
        self.model_output_names = [output.name for output in self.onnx_model.graph.output]
        self.model_name = result.split("/")[-1].split(".")[0]
        log_info(">> Model exported successfully.")
        log_info(f"  - Model Name: {self.model_name}")
        log_info(f"  - Model Input Shape: {self.model_input_shape}")
        log_info(f"  - Output Names: {self.model_output_names}")
        log_info("-------------------------------------------------------------------\n")
        return result  # Return the path to the exported ONNX model
    
    def __create_export_dir(self):
        """
        Create the export directory if it does not exist.
        @warning: Before calling this, one should call __load_config() and __export_onnx() to set the model_name.
        - models/
            - <model_name>/
                - mlir/
                - bmodel/
        """
        dir = os.path.join(self.model_export_dir, self.model_name)
        log_info(f">> Creating export directories...")
        try:
            # create the main export directory: models/<model_name>
            if not os.path.exists(dir): os.makedirs(dir)
            self.model_export_dir = dir 
            log_info(f"  - model_export_dir: {self.model_export_dir}")
            # create subdirectories for mlir 
            self.mlir_export_dir = os.path.join(dir, self.mlir_export_dir)
            if not os.path.exists(self.mlir_export_dir): os.makedirs(self.mlir_export_dir)
            log_info(f"  - mlir_export_dir: {self.mlir_export_dir}")
            # create subdirectories for bmodel
            self.bmodel_export_dir = os.path.join(dir, self.bmodel_export_dir)
            if not os.path.exists(self.bmodel_export_dir): os.makedirs(self.bmodel_export_dir)
            log_info(f"  - bmodel_export_dir: {self.bmodel_export_dir}\n")
        except Exception as e:
            log_error(f">> Failed to create export directory `{dir}`: {e}")
            sys.exit(1)

    def __move_file(self, src, dst_dir):
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst):
            os.remove(dst)  # Remove existing file
        shutil.move(src, dst)


    def __move_models_to_export_dir(self):
        """
        Move the downloaded PyTorch model and exported ONNX model to the export directory.
        """
        # helper function to move a file and replace it if it already exists
        
            
        onnx_model_path = f"{self.model_name}.onnx"
        pt_model_path = f"{self.model_name}.pt"
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model file {onnx_model_path} does not exist.")
        if not os.path.exists(pt_model_path):
            raise FileNotFoundError(f"PyTorch model file {pt_model_path} does not exist.")
        try:
            # move PyTorch model to model_export_dir
            pt_model_dest = self.model_export_dir
            if not os.path.exists(pt_model_dest):
                os.makedirs(pt_model_dest)
            self.__move_file(pt_model_path, pt_model_dest)
            self.__move_file(onnx_model_path, pt_model_dest)
            self.pt_model_path = os.path.join(pt_model_dest, pt_model_path)
            self.onnx_model_path = os.path.join(pt_model_dest, onnx_model_path)
            log_info(f">> PyTorch and ONNX models moved to {pt_model_dest}\n")
        except Exception as e:
            log_error(f">> Failed to move ONNX model: {e}")
            sys.exit(1)


    def export_mlir(self, onnx_model_path: str):
        """
        Function to export the ONNX model to MLIR format.
        @param onnx_model_path: Path to the ONNX model file.
        @note: This function saves the results to the mlir_export_dir directory (models/<model_name>/mlir).
        """
        log_info("==================================================================")
        log_info(">> Transforming ONNX model to MLIR format...")
        if not os.path.exists(onnx_model_path):
            log_error(f"ONNX model file {onnx_model_path} does not exist.")
            sys.exit(1)
        model_name = self.model_name
        model_def = onnx_model_path
        input_shape = str([list(self.model_input_shape)] if self.model_input_shape else [])
        mean = ", ".join(str(v) for v in self.img_ch_mean)
        scale = ", ".join(str(v) for v in self.img_ch_scale)
        keep_aspect_ratio = self.keep_aspect_ratio
        pixel_format = self.pixel_format
        output_names = ", ".join(self.model_output_names)
        test_input = self.mlir_test_input
       
        # --model_name yolov5s \
        # --model_def ../yolov5s.onnx \
        # --input_shapes [[1,3,640,640]] \
        # --mean 0.0,0.0,0.0 \
        # --scale 0.0039216,0.0039216,0.0039216 \
        # --keep_aspect_ratio \
        # --pixel_format rgb \
        # --output_names 350,498,646 \
        # --test_input ../image/dog.jpg \
        # --test_result yolov5s_top_outputs.npz \
        # --mlir yolov5s.mlir
        
        test_result_path= os.path.join(self.mlir_export_dir, self.model_name + "_top_outputs.npz")
        mlir_output_path = os.path.join(self.mlir_export_dir, f"{model_name}.mlir")
        self.mlir_model_path = mlir_output_path  # Update the mlir_model_path attribute
        args = [
            "--model_name", model_name,
            "--model_def", model_def,
            "--input_shapes", input_shape,
            "--mean", str(mean),
            "--scale", str(scale),
            "--pixel_format", pixel_format,
            "--output_names", output_names,
            "--test_result", test_result_path,
            "--mlir", mlir_output_path,

        ]
        if keep_aspect_ratio:
            args += ["--keep_aspect_ratio"]
        if os.path.exists(test_input):
            args += ["--test_input", test_input]
        else:
            log_error(f">> Test input file {test_input} does not exist. Skipping test input.")
        
        log_info(">> Running `model_transform.py` with the following arguments:")
        i = 0
        while i < len(args):
            if args[i].startswith("--"):
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    log_info(f"    {args[i]} {args[i+1]}")
                    i += 2
                else:
                    log_info(f"    {args[i]}")
                    i += 1
            else:
                log_info(f"    {args[i]}")
                i += 1
        try:
            subprocess.run(
                ["model_transform"] + args,
                check=True,
            )
            ## move all the files in current dir start with model_name to mlir_export_dir
            # for file in os.listdir("."):
            #     if file.startswith(model_name):
            #         self.__move_file(file, self.mlir_export_dir)
            self.mlir_model_path = mlir_output_path  # Update the mlir_model_path attribute
            log_info(f">> MLIR file saved to {mlir_output_path}")
            log_info(f">> Test results saved to {test_result_path}")

        except Exception as e:
            log_error(f">> Failed to transform ONNX model to MLIR format: {e}")
            sys.exit(1)

        log_info(">> ONNX model transformed to MLIR format successfully.")
        log_info("-------------------------------------------------------------------\n")

    def __calibrate_model(self) -> str:
        """
        Function to calibrate the model for INT8 quantization.
        This function should be called before exporting to bmodel format.
        @return: Path to the calibration table file.
        @note: This function uses the calibration dataset and input number specified in the config file.
        """
        log_info("==================================================================")
        log_info(">> Calibrating model for INT8 quantization...")
        if self.bmodel_quantize != "INT8":
            log_info(">> Skipping calibration as the quantization method is not INT8.")
            return
        if not os.path.exists(self.calib_dataset):
            log_error(f">> Calibration dataset path `{self.calib_dataset}` does not exist. Please provide a valid dataset path for calibration.")
            sys.exit(1)
        if self.calib_input_num <= 0:
            log_error(f">> Invalid calibration input number `{self.calib_input_num}`. It should be a positive integer.")
            sys.exit(1)
        ### run_calibration.py yolov5s.mlir \
        ###     --dataset <path/to/coco/dataset> \
        ###     --input_num <calib_input_num> \
        ###     -o <model_name>_cali_table
        output_path = f"{self.model_name}_cali_table"
        args = [
            "--dataset", self.calib_dataset,
            "--input_num", str(self.calib_input_num),
            "-o", output_path
        ]
        try:
            subprocess.run(
                ["run_calibration.py", self.mlir_model_path] + args,
                check=True,
            )
            log_info(f">> Calibration table saved to {output_path}")
        except Exception as e:
            log_error(f">> Failed to run calibration script: {e}")
            sys.exit(1)
        return output_path  # Return the path to the calibration table
        
    
    def export_bmodel(self, mlir_model_path: str):
        """
        Function to export the MLIR model to bmodel format.
        @param mlir_model_path: Path to the MLIR model file.
        @note: This function saves the results to the bmodel_export_dir directory (models/<model_name>/bmodel).
        """
        log_info("==================================================================")
        log_info(">> Deploying MLIR model to bmodel format...")
        if not os.path.exists(mlir_model_path):
            log_error(f"MLIR model file {mlir_model_path} does not exist.")
            sys.exit(1)
        
        ## args for Fxx quantization
        # --mlir yolov5s.mlir \
        # --quantize F16 \
        # --processor bm1684x \
        # --test_input yolov5s_in_f32.npz \
        # --test_reference yolov5s_top_outputs.npz \
        # --model yolov5s_1684x_f16.bmodel

        # args for INT8 quantization
        # --mlir yolov5s.mlir \
        # --quantize INT8 \
        # --calibration_table yolov5s_cali_table \
        # --processor bm1684x \
        # --test_input yolov5s_in_f32.npz \
        # --test_reference yolov5s_top_outputs.npz \
        # --tolerance 0.85,0.45 \
        # --model yolov5s_1684x_int8.bmodel

        # Prepare arguments for model_deploy

        args = []
        if self.bmodel_quantize == "INT8":
            # calibrate the model first
            calib_table_path = self.__calibrate_model()
            if not os.path.exists(calib_table_path):
                log_error(f">> Calibration table {calib_table_path} does not exist.")
                sys.exit(1)
            # set the arguments for INT8 quantized bmodel deployment
            args = [
                "--mlir", mlir_model_path,
                "--quantize", self.bmodel_quantize,
                "--calibration_table", calib_table_path,
                "--processor", self.bmodel_processor,
                "--test_input", f"{self.model_name}_in_f32.npz",
                "--test_reference", os.path.join(self.mlir_export_dir, f"{self.model_name}_top_outputs.npz"),
                "--model", os.path.join(self.bmodel_export_dir, f"{self.model_name}_{self.bmodel_processor}_int8.bmodel"),
            ]
            if self.bmodel_tolerance:
                args += ["--tolerance", f"{self.bmodel_tolerance[0]},{self.bmodel_tolerance[1]}"]
            # deploy the model to bmodel format
            try:
                subprocess.run(
                    ["model_deploy.py"] + args,
                    check=True,
                )
                log_info(f">> Bmodel file saved to {self.bmodel_export_dir}")
            except Exception as e:
                log_error(f">> Failed to deploy MLIR model to bmodel format: {e}")
                sys.exit(1)
        
        else:
            # set the arguments for Fxx quantized bmodel deployment
            args = [
                "--mlir", mlir_model_path,
                "--quantize", self.bmodel_quantize,
                "--processor", self.bmodel_processor,
                "--test_input", f"{self.model_name}_in_f32.npz",
                "--test_reference", os.path.join(self.mlir_export_dir, f"{self.model_name}_top_outputs.npz"),
                "--model", os.path.join(self.bmodel_export_dir, f"{self.model_name}_{self.bmodel_processor}_{self.bmodel_quantize.lower()}.bmodel")
            ]
            # deploy the model to bmodel format
            try:
                subprocess.run(
                    ["model_deploy.py"] + args,
                    check=True,
                )
                log_info(f">> Bmodel file saved to {self.bmodel_export_dir}")
            except Exception as e:
                log_error(f">> Failed to deploy MLIR model to bmodel format: {e}")
                sys.exit(1)

        self.bmodel_path = os.path.join(self.bmodel_export_dir, f"{self.model_name}_{self.bmodel_processor}_{self.bmodel_quantize.lower()}.bmodel")
        log_info(">> MLIR model deployed to bmodel format successfully.")
        log_info(f">> Bmodel file path: {self.bmodel_path}")
        log_info("--------------------------------------------------------------------")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to Bmodel format.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for model export.",
    )
    args = parser.parse_args()
    exporter = YOLOBModelExporter(args.config)
    exporter.export_mlir(exporter.onnx_model_path)
    exporter.export_bmodel(exporter.mlir_model_path)  
