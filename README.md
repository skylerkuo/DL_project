# DL_project

BERT 部份:

先跑 train_teacher.py

再跑 distill.py

Bert的所有模型都能夠用test.py 測試效果



LLM 部份:

qwen_classification.py: 用qwen做分類的範例程式 測試的次數和模型可以自己改

finetune__qwen.py: 用情感分類的資料並使用QLORA去微調qwen (這個程式只要改模型名稱就可以去微調很多不同的模型)

combine_qwen.py: 將用finetune__qwen.py訓練好的adapter和模型本體合併成一個完整的模型 合併之後可以再用qwen_classification.py去測試 只需修改qwen_classification.py的模型名稱
