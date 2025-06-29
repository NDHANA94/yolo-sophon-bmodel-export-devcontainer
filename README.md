## VSCode devcontainer to generate YOLO BModels for Sophon TPU/NPU 

This repository provides a VSCode devcontainer setup for generating **YOLO BModels** using **Sophon** tools. It includes all necessary dependencies and configurations to streamline the process.

</br>

### 🔷 Features
- VSCode devcontainer setup for easy development
- Pre-configured environment for BModel generation using [tpu-mlir](https://github.com/sophgo/tpu-mlir.git)
- Easy model conversion with [yaml configuration](yolo_bmodel_exporter/bmodel_export_config.yaml)
- Support for various model quantizations (F32, BF16, F16, INT8)
- Support different Sophon chips (see [Supported Chip and Quantization Options](#🔷-supported-chip-and-quantization-options))

</br>

### 🔷 Supported Chip and Quantization Options

| Chip Name | F32 | F16 | BF16 | INT8 |
|-----------|-----|-----|------|------|
| bm1684x   | ✅  | ✅  | ✅   | ❌   |
| bm1684    | ✅  | ❌  | ❌   | ❌   |
| bm1688    | ✅  | ✅  | ✅   | ❌   |
| cv186x    | ✅  | ✅  | ✅   | ❌   |
| bm1690    | ✅  | ✅  | ✅   | ❌   |

> **Note:** INT8 quantization is not completed successfully yet due to an error shown [here](#int8-quantization-error)

</br>

### 🔷 Usage

1. **Clone the repository:**
    ```bash
    git clone git@github.com:NDHANA94/yolo-sophon-bmodel-export-devcontainer.git  
    ```

2. **Open the repository in VSCode:**
    ```bash
    cd yolo-sophon-bmodel-export-devcontainer
    code .
    ```
3. **Open workspace in container**
    - Click on the **blue button** in the bottom left corner of VSCode.
    - Select **"Reopen in Container"**. This will build the container and open the workspace inside it.

4. **Configure BModel export settings:**
    - Open [bmodel_export_config.yaml](yolo_bmodel_exporter/bmodel_export_config.yaml) in the [yolo_bmodel_exporter](file://yolo_bmodel_exporter/) directory.
    - Modify the configuration parameters as needed.

5. **Export BModel:**
    - Open a terminal in VSCode.
    - Follow the steps below to export the BModel:
      ```bash
      cd yolo_bmodel_exporter
      mkdir ws && cd ws
      python3 ../export_yolo_bmodel.py --config ../bmodel_export_config.yaml 
      ```
      > This will save the generated BModel in the `ws/models/<model_name>/bmodel/` directory.



---

</br></br>

### `INT8` Quantization Error:

```bash
...
[Running]: npz_tool.py compare yolo11n_bm1688_int8_sym_tpu_outputs.npz models/yolo11n/mlir/yolo11n_top_outputs.npz --tolerance 0.8,0.5 --except - -vv
compare output0_Concat:   0%|                                                                                                                      | 0/1 [00:00<?, ?it/s]

[output0_Concat                  ]  NOT_SIMLIAR [FAILED]
    (1, 84, 8400) float32 
    cosine_similarity      = 0.759107
    euclidean_similarity   = -0.614530
    sqnr_similarity        = -16.145945
top-k:
 idx-t  target  idx-r  ref
  20789 39.81035 5599 637.37683
  20790 39.81035 5519 637.02155
  20785 39.81035 879 636.9863
  20786 39.81035 959 636.9486
  20787 39.81035 5679 636.9191
  20788 39.81035 5759 636.51587
  20783 39.81035 3839 636.4876
  20796 39.81035 5439 636.43823
  20794 39.81035 799 636.362
  20782 39.81035 559 636.3137
1 compared
0 passed
  0 equal, 0 close, 0 similar
1 failed
  0 not equal, 1 not similar
min_similiarity = (0.7591071724891663, -0.6145298527734675, -16.145944595336914)
Target    yolo11n_bm1688_int8_sym_tpu_outputs.npz
Reference models/yolo11n/mlir/yolo11n_top_outputs.npz
npz compare FAILED.
compare output0_Concat: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.52it/s]
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/tools/model_deploy.py", line 570, in <module>
    lowering_patterns = tool.lowering()
  File "/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/tools/model_deploy.py", line 228, in lowering
    self.validate_tpu_mlir()
  File "/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/tools/model_deploy.py", line 336, in validate_tpu_mlir
    f32_blobs_compare(self.tpu_npz, self.ref_npz, self.tolerance, self.excepts, fuzzy_match=self.fazzy_match)
  File "/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/utils/mlir_shell.py", line 1028, in f32_blobs_compare
    _os_system(cmd, log_level=log_level)
  File "/usr/local/lib/python3.10/dist-packages/tpu_mlir/python/utils/mlir_shell.py", line 419, in _os_system
    raise RuntimeError("[!Error]: {}".format(cmd_str))
RuntimeError: [!Error]: npz_tool.py compare yolo11n_bm1688_int8_sym_tpu_outputs.npz models/yolo11n/mlir/yolo11n_top_outputs.npz --tolerance 0.8,0.5 --except - -vv
>> Failed to deploy MLIR model to bmodel format: Command '['model_deploy', '--mlir', 'models/yolo11n/mlir/yolo11n.mlir', '--quantize', 'INT8', '--calibration_table', 'yolo11n_cali_table', '--processor', 'bm1688', '--test_input', 'yolo11n_in_f32.npz', '--test_reference', 'models/yolo11n/mlir/yolo11n_top_outputs.npz', '--model', 'models/yolo11n/bmodel/yolo11n_bm1684x_int8.bmodel']' returned non-zero exit status 1.
```

---

</br>

> **Author:** WM Nipun Dhananjaya Weerakkodi  | [nipun.dhananjaya@gmail.com](mailto:nipun.dhananjaya@gmail.com)