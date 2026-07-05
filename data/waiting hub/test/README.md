# Waiting Hub Test

本目录只负责调用 `Real_data/train` 产出的微调模型，对 `data/waiting hub/候选库4万.xlsx` 做推理。

## 目录作用

- `predict_waiting_hub.py`
  - 默认读取 `data/waiting hub/候选库4万.xlsx`
  - 默认读取 `Real_data/train/latest_run.json` 指向的最新微调结果
  - 支持选择 `best` 或 `final` checkpoint
  - 输出列固定为：
    - `Combo`
    - `SMILES`
    - `Prediction`
- `predictions/`
  - 保存每次推理生成的 Excel 和 CSV。
- `reports/`
  - 保存每次推理的摘要信息。
- `latest_inference.json`
  - 指向最近一次推理结果。

## 默认前提

先在：

```text
E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\Real_data\train
```

完成一次正式微调。训练成功后会自动生成：

```text
Real_data/train/latest_run.json
```

如果没有这个文件，就说明还没有可直接调用的微调模型。

## 建议先检查输入和训练元数据

这个命令不加载大模型，只检查：

- 候选库 Excel 是否存在
- 列名是否包含 `Combo` 和 `SMILES`
- 最新训练运行是否存在

```powershell
cd "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\data\waiting hub\test"
python predict_waiting_hub.py --preview-only
```

## 使用最新 best 模型推理

```powershell
cd "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\data\waiting hub\test"
python predict_waiting_hub.py ^
  --checkpoint best ^
  --device cuda:0 ^
  --batch-size 8
```

## 使用最新 final 模型推理

```powershell
cd "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\data\waiting hub\test"
python predict_waiting_hub.py ^
  --checkpoint final ^
  --device cuda:0 ^
  --batch-size 8
```

## 指定某一次训练运行目录推理

如果你不想用最新一次运行，可以显式指定：

```powershell
cd "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\data\waiting hub\test"
python predict_waiting_hub.py ^
  --run-dir "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\Real_data\train\runs\real_data_head_log2_YYYYMMDD_HHMMSS" ^
  --checkpoint best ^
  --device cuda:0 ^
  --batch-size 8
```

## 推理输出位置

每次推理都会创建一个独立目录：

```text
predictions/<run_name>_<checkpoint>_<timestamp>/
├─ waiting_hub_predictions.xlsx
└─ waiting_hub_predictions.csv
```

对应摘要文件在：

```text
reports/<run_name>_<checkpoint>_<timestamp>/summary.json
```

输出表格列名固定为：

```text
Combo | SMILES | Prediction
```

## 依赖提醒

当前环境实测缺少 `transformers`，如果微调时用了 LoRA，还需要 `peft`：

```powershell
cd "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP"
pip install -r requirements.txt
pip install openpyxl
pip install peft
```

如果你使用的不是 LoRA 模型，最后一条 `pip install peft` 可以省略。

## 查看帮助

```powershell
python predict_waiting_hub.py --help
```
