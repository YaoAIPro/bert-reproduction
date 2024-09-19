@echo off

REM 设置 BERT 和 GLUE 目录
SET BERT_BASE_DIR=E:\Learning\Model\Google\BERT\multi_cased_L-12_H-768_A-12
SET DATA_DIR=E:\Personal\git-project\bert-reproduction\dataset
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
python ../create_pretraining_data.py ^
  --input_file=%DATA_DIR%\origin_data\%Dataset%.txt ^
  --output_file=%DATA_DIR%\pretraining_data\%Dataset%.tfrecord ^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt ^
  --do_lower_case=True ^
  --max_seq_length=128 ^
  --max_predictions_per_seq=20 ^
  --masked_lm_prob=0.15 ^
  --random_seed=12345 ^
  --dupe_factor=5

REM 检查Python脚本的返回代码
if %ERRORLEVEL% neq 0 (
    echo Python script execution failed.
    pause
    exit /b %ERRORLEVEL%
)

REM 如果一切成功，输出完成信息
echo Script executed successfully.
pause
