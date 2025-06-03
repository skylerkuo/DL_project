# DL_project

BERT 部份:

先跑 train_teacher.py

再跑 distill.py

Bert的所有模型都能夠用test.py 測試效果



LLM 部份:

finetune__qwen.py: 用情感分類的資料並使用QLORA去微調qwen

qwen_classification.py: 測試個模型的表現

distrill_qwen.py: 蒸餾語言模型
