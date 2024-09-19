@echo off

REM 设置 BERT 和 GLUE 目录
SET BERT_BASE_DIR=E:\Learning\Model\Google\BERT\multi_cased_L-12_H-768_A-12
SET BASE_DIR=E:\Learning\Algorithm-Reproduction\bert
SET Dataset=darkmyth

REM 激活虚拟环境
call conda activate bert

REM 检查虚拟环境是否激活成功
if %ERRORLEVEL% neq 0 (
    echo Failed to activate conda environment 'bert'.
    pause
    exit /b %ERRORLEVEL%
)

REM 输出确认虚拟环境已激活
echo Activated conda environment 'bert'.

REM 运行 Python 脚本
python ../run_pretraining.py ^
  --input_file=%BASE_DIR%\dataset\pretraining_data\%Dataset%.tfrecord  ^
  --output_dir=%BASE_DIR%\output\pretraining_%Dataset%  ^
  --do_train=True ^
  --do_eval=True ^
  --bert_config_file=%BERT_BASE_DIR%\bert_config.json ^
  --init_checkpoint=%BERT_BASE_DIR%\bert_model.ckpt ^
  --train_batch_size=32 ^
  --max_seq_length=128 ^
  --max_predictions_per_seq=20 ^
  --num_train_steps=20 ^
  --num_warmup_steps=10 ^
  --learning_rate=2e-5

REM 检查Python脚本的返回代码
if %ERRORLEVEL% neq 0 (
    echo Python script execution failed.
    pause
    exit /b %ERRORLEVEL%
)

REM 如果一切成功，输出完成信息
echo Script executed successfully.
pause
