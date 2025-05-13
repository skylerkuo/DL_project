# DL_project

BERT 部份:

先跑train_teacher.py

再跑 distill.py

Bert的所有模型都能夠用test.py 測試效果



LLM 部份:

qwen_classification.py: 用qwen做分類的範例程式 (這個程式只要改模型名稱就可以去微調很多不同的模型)

finetune__qwen.py: 用情感分類的資料並使用QLORA去微調qwen
