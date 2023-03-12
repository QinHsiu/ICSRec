# ICSRec

This is our Pytorch implementation for the paper: "**Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation**".

## Environment  Requirement

* Pytorch>=1.7.0
* Python>=3.7 

## Model Overview

 ![avator](./pics/model.png)

## Usage

Please run the following command to install all the requirements:  

```python
pip install -r requirements.txt
```

## Evaluate Model

We provide the trained models on Beauty, Sports_and_Outdoors, Toys_and_Games and ML-1M datasets in `./src/output/<Data_name>`folder. You can directly evaluate the trained models on test set by running:

```
python main.py --data_name <Data_name> --model_idx 0 --do_eval --encoder SAS/GRU
```

On Beauty:

```python
python main.py --data_name Beauty --model_idx 0 --do_eval --encoder SAS
```

```
{'Epoch': 0, 'HIT@5': '0.0698', 'NDCG@5': '0.0494', 'HIT@10': '0.0959', 'NDCG@10': '0.0578', 'HIT@20': '0.1298', 'NDCG@20': '0.0663'}
```

```
python main.py --data_name Beauty --model_idx 0 --do_eval --encoder GRU
```

```
{'Epoch': 0, 'HIT@5': '0.0515', 'NDCG@5': '0.0365', 'HIT@10': '0.0740', 'NDCG@10': '0.0437', 'HIT@20': '0.1014', 'NDCG@20': '0.0506'}
```

On Sports_and_Outdoors:

```python
python main.py --data_name Sports_and_Outdoors --model_idx 0 --do_eval --encoder SAS
```

```
{'Epoch': 0, 'HIT@5': '0.0403', 'NDCG@5': '0.0283', 'HIT@10': '0.0565', 'NDCG@10': '0.0335', 'HIT@20': '0.0794', 'NDCG@20': '0.0393'}
```

```
python main.py --data_name Sports_and_Outdoors --model_idx 0 --do_eval --encoder GRU
```

```
{'Epoch': 0, 'HIT@5': '0.0278', 'NDCG@5': '0.0191', 'HIT@10': '0.0404', 'NDCG@10': '0.0232', 'HIT@20': '0.0596', 'NDCG@20': '0.0280'}
```

On Toys_and_Games:

```python
python main.py --data_name Toys_and_Games --model_idx 0 --do_eval --encoder SAS
```

```
{'Epoch': 0, 'HIT@5': '0.0788', 'NDCG@5': '0.0571', 'HIT@10': '0.1055', 'NDCG@10': '0.0657', 'HIT@20': '0.1368', 'NDCG@20': '0.0736'}
```

```
python main.py --data_name Toys_and_Games --model_idx 0 --do_eval --encoder GRU
```

```
{'Epoch': 0, 'HIT@5': '0.0519', 'NDCG@5': '0.0388', 'HIT@10': '0.0699', 'NDCG@10': '0.0446', 'HIT@20': '0.0950', 'NDCG@20': '0.0509'}
```

On ML-1M:

```python
python main.py --data_name ml-1m --model_idx 0 --do_eval --encoder SAS
```

```
{'Epoch': 0, 'HIT@5': '0.2442', 'NDCG@5': '0.1708', 'HIT@10': '0.3369', 'NDCG@10': '0.2007', 'HIT@20': '0.4518', 'NDCG@20': '0.2297'}
```

```
python main.py --data_name ml-1m --model_idx 0 --do_eval --encoder GRU
```

```
{'Epoch': 0, 'HIT@5': '0.2033', 'NDCG@5': '0.1398', 'HIT@10': '0.2889', 'NDCG@10': '0.1673', 'HIT@20': '0.4045', 'NDCG@20': '0.1964'}
```



## Train Model

Please train the model using the Python script `main.py`.

You can run the following command to train the model on Beauty datasets:

```
python main.py --data_name Beauty --rec_weight 0.1 --lambda_0 0.1 --beta_0 0.1 --f_neg --intent_num 512 
```

## Overall Performances

N represents Normalized Discounted Cumulative Gain(NDCG) and H represents Hit Ratio (HR).

| Dataset | Metrc | BPR    | GRU4Rec | Caser  | SASRec | BERT4Rec | S3Rec  | CL4SRec | CoSeRec  | DuoRec   | ICLRec   | DSSRec | SINE   | ICSRec     | Improv. |
| ------- | ----- | ------ | ------- | ------ | ------ | -------- | ------ | ------- | -------- | -------- | -------- | ------ | ------ | ---------- | ------- |
| Sports  | H@5   | 0.0123 | 0.0162  | 0.0154 | 0.0214 | 0.0217   | 0.0121 | 0.0231  | 0.0290   | *0.0312* | 0.0290   | 0.0209 | 0.0240 | **0.0403** | 29.17%  |
| Sports  | H@20  | 0.0369 | 0.0421  | 0.0399 | 0.0500 | 0.0604   | 0.0344 | 0.0557  | 0.0636   | *0.0696* | 0.0646   | 0.0499 | 0.0610 | **0.0794** | 14.10%  |
| Sports  | N@5   | 0.0076 | 0.0103  | 0.0114 | 0.0144 | 0.0143   | 0.0084 | 0.0146  | *0.0196* | 0.0195   | 0.0191   | 0.0139 | 0.0152 | **0.0283** | 44.39%  |
| Sports  | N@20  | 0.0144 | 0.0186  | 0.0178 | 0.0224 | 0.0251   | 0.0146 | 0.0238  | 0.0293   | *0.0302* | 0.0291   | 0.0221 | 0.0255 | **0.0393** | 30.13%  |
| Beauty  | H@5   | 0.0178 | 0.0180  | 0.0251 | 0.0377 | 0.0360   | 0.0189 | 0.0401  | 0.0504   | *0.0561* | 0.0500   | 0.0408 | 0.0354 | **0.0698** | 24.42%  |
| Beauty  | H@20  | 0.0474 | 0.0427  | 0.0643 | 0.0894 | 0.0984   | 0.0487 | 0.0974  | 0.1034   | *0.1228* | 0.1058   | 0.0894 | 0.0964 | **0.1298** | 5.70%   |
| Beauty  | N@5   | 0.0109 | 0.0116  | 0.0145 | 0.0241 | 0.0216   | 0.0115 | 0.0268  | 0.0339   | *0.0348* | 0.0326   | 0.0263 | 0.0213 | **0.0494** | 41.95%  |
| Beauty  | N@20  | 0.0192 | 0.0186  | 0.0298 | 0.0386 | 0.0391   | 0.0198 | 0.0428  | 0.0487   | *0.0536* | 0.0483   | 0.0399 | 0.0384 | **0.0663** | 23.69%  |
| Toys    | H@5   | 0.0122 | 0.0121  | 0.0205 | 0.0429 | 0.0371   | 0.0456 | 0.0503  | 0.0533   | *0.0655* | 0.0597   | 0.0447 | 0.0385 | **0.0788** | 20.30%  |
| Toys    | H@20  | 0.0327 | 0.0290  | 0.0542 | 0.0957 | 0.0760   | 0.0940 | 0.0990  | 0.1037   | *0.1293* | 0.1139   | 0.0942 | 0.0957 | **0.1368** | 5.80%   |
| Toys    | N@5   | 0.0076 | 0.0077  | 0.0125 | 0.0245 | 0.0259   | 0.0314 | 0.0264  | 0.0370   | 0.0392   | *0.0404* | 0.0297 | 0.0225 | **0.0571** | 41.34%  |
| Toys    | N@20  | 0.0132 | 0.0123  | 0.0221 | 0.0397 | 0.0368   | 0.0452 | 0.0404  | 0.0513   | *0.0574* | 0.0557   | 0.0437 | 0.0386 | **0.0736** | 28.22%  |
| ML-1M   | H@5   | 0.0247 | 0.0806  | 0.0912 | 0.1078 | 0.1308   | 0.1078 | 0.1142  | 0.1128   | *0.2098* | 0.1382   | 0.1371 | 0.0990 | **0.2445** | 16.54%  |
| ML-1M   | H@20  | 0.0750 | 0.2081  | 0.2228 | 0.2745 | 0.3354   | 0.3114 | 0.2818  | 0.2950   | *0.4098* | 0.3368   | 0.3275 | 0.2705 | **0.4518** | 10.25%  |
| ML-1M   | N@5   | 0.0159 | 0.0475  | 0.0565 | 0.0681 | 0.0804   | 0.0616 | 0.0705  | 0.0692   | *0.1433* | 0.0889   | 0.0898 | 0.0586 | **0.1710** | 19.33%  |
| ML-1M   | N@20  | 0.0297 | 0.0834  | 0.0931 | 0.1156 | 0.1384   | 0.1204 | 0.1170  | 0.1247   | *0.2007* | 0.1450   | 0.1440 | 0.1066 | **0.2297** | 14.45%  |
