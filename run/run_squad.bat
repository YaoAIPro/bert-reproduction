@echo off

REM 设置 BERT 和 GLUE 目录
SET BERT_BASE_DIR=E:\Learning\Model\Google\BERT\multi_cased_L-12_H-768_A-12
SET SQUAD_DIR=E:\Personal\git-project\bert-reproduction\dataset\SQuAD 1.1
SET Dataset=SQuAD1-1

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
python ../run_squad.py ^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%\bert_config.json ^
  --init_checkpoint=%BERT_BASE_DIR%\bert_model.ckpt ^
  --do_train=True ^
  --train_file=%SQUAD_DIR%\train-v1.1.json ^
  --do_predict=True ^
  --predict_file=%SQUAD_DIR%\dev-v1.1.json ^
  --train_batch_size=12 ^
  --learning_rate=3e-5 ^
  --num_train_epochs=2.0 ^
  --max_seq_length=384 ^
  --doc_stride=128 ^
  --output_dir=E:\Learning\Algorithm-Reproduction\bert\output\%Dataset%


REM 检查Python脚本的返回代码
if %ERRORLEVEL% neq 0 (
    echo Python script execution failed.
    pause
    exit /b %ERRORLEVEL%
)

REM 如果一切成功，输出完成信息
echo Script executed successfully.
pause
