# Real Data Train

本目录只负责 `Real_data/real_data_smiles_target.xlsx` 的微调流程，不改动项目根目录已有训练脚本。

## 目录作用

- `train_real_data.py`
  - 调用预训练模型和 tokenizer。
  - 读取 `Real_data/real_data_smiles_target.xlsx`。
  - 默认使用 `log2` 目标变换，也支持 `log10`、`normalize`、`none`。
  - 支持两种微调方式：
    - `head`：默认预测头微调
    - `lora`：LoRA 低秩微调
- `configs/`
  - 保存每次运行的参数配置。
- `prepared_data/`
  - 保存清洗后的 `train/val/test` 拆分结果。
- `runs/`
  - 保存每次微调运行的模型、报告、预测结果。
- `reports/`
  - 可放人工整理的总报告。
- `latest_run.json`
  - 成功训练后会自动更新，供 `data/waiting hub/test/` 直接调用。

## 默认输入

- 训练数据：`E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\Real_data\real_data_smiles_target.xlsx`
- 预训练模型：`E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\models\qwen_1.8b_smiles_pretrained\final_model`
- tokenizer：`E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\models\qwen_1.8b_smiles_pretrained\tokenizer.json`

注意：

- 你原始目标里写的是 `.xls`，但当前仓库实际文件是 `.xlsx`。
- 当前环境实测缺少 `transformers` 和 `peft`，正式训练前请先安装依赖。

## 建议先执行

```powershell
cd "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP"
pip install -r requirements.txt
pip install openpyxl
```

如果要使用 LoRA，再执行：

```powershell
pip install peft
```

## 第 1 步：只检查数据并生成拆分文件

这一步不加载大模型，适合先确认 Excel、列名、拆分结果是否正确。

```powershell
cd "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\Real_data\train"
python train_real_data.py --prepare-only
```

执行后重点查看：

- `prepared_data/<run_name>/train.xlsx`
- `prepared_data/<run_name>/val.xlsx`
- `prepared_data/<run_name>/test.xlsx`
- `configs/<run_name>.json`

## 第 2 步：默认预测头微调

默认目标变换是 `log2`。

```powershell
cd "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\Real_data\train"
python train_real_data.py ^
  --finetune-method head ^
  --target-transform log2 ^
  --num-epochs 5 ^
  --batch-size 1 ^
  --gradient-accumulation-steps 4 ^
  --learning-rate 1e-5 ^
  --device cuda:0
```

如果希望训练新扩展的 embedding，可加：

```powershell
--train-embeddings
```

## 第 3 步：LoRA 微调

```powershell
cd "E:\School Work\Deep Learning\Paper\Huaxi\virtual_LNP\Real_data\train"
python train_real_data.py ^
  --finetune-method lora ^
  --target-transform log2 ^
  --num-epochs 5 ^
  --batch-size 1 ^
  --gradient-accumulation-steps 4 ^
  --learning-rate 1e-5 ^
  --device cuda:0 ^
  --lora-r 8 ^
  --lora-alpha 16 ^
  --lora-dropout 0.05
```

## 可选目标变换

默认：

```powershell
--target-transform log2
```

可改为：

```powershell
--target-transform log10
```

或：

```powershell
--target-transform normalize
```

如果要直接用原始值：

```powershell
--target-transform none
```

## 训练完成后会生成

每次运行都会生成一个独立目录：

```text
runs/<run_name>/
├─ checkpoints/
│  ├─ best/
│  └─ final/
├─ reports/
│  ├─ metrics_history.csv
│  ├─ best_val_predictions.xlsx
│  ├─ best_test_predictions.xlsx
│  ├─ final_val_predictions.xlsx
│  ├─ final_test_predictions.xlsx
│  └─ summary.json
└─ run_metadata.json
```

其中：

- `best/` 是验证集 RMSE 最优模型
- `final/` 是最后一个 epoch 的模型
- `latest_run.json` 会指向最近一次成功训练的运行目录

## 查看帮助

```powershell
python train_real_data.py --help
```
