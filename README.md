# Drama2Vec
テレビドラマのデータを使って、Item2Vecやってみた。  
詳細は、Item2VecTrial.pdf参照  

## Dataset  
https://www.jstage.jst.go.jp/article/itej/70/11/70_J255/_pdf  
福島悠介, 山崎俊彦, 相澤清晴, 放送前の情報のみを用いたテレビドラマの視聴率予測, 映像情報メディア学会誌 Vol.70, No.11, pp.J255-J261  
が提供している2008/4~2015/6までのドラマ678本をベースにした。

## Directory
-CBOWresult: CBOW形式で学習した学習モデル  
-data: 上記Paper提供のデータセット  
-data_forExcel: 上記Paper提供のデータセット(Excel確認用)  
-DataID.csv: 使用したデータ項目とそのIDナンバー   
　　18, 19, ..., 26: 放送時間帯  
　　under5, 5~10, ..., 20~25, over25: 平均視聴率  
-Item2VecTrial.pdf: 説明スライド  
-data_for_calc.csv: 上記Paper提供のデータセットを今回の計算用に編集したもの  
-drama2vec.py: 実行ファイル  

## Prerequisites
-python2.7  
-tensorflow  
-prettyprint  
-numpy  
-pandas  
-matplotlib  
-sklearn  

## Usage
### Similarity
指定のデータ項目の意味合いの近いものを出力する。  

1. DataID.csvより、確認したい項目のIDを記憶する。(例:木村拓哉=297)
2. 以下を実行  
`$ python drama2vec.py —-mode=similarity —-valid_id=ID_NUMBER`  
  
__例:木村拓哉の場合__  
`$ python drama2vec.py --mode=similarity --valid_id=297`
  
Output    
```  
--evaluating as similarity--  
"Nearest to 木村拓哉"  
----  
"水谷豊    cosine value :0.362201"
"北大路欣也    cosine value :0.350494"
"over25    cosine value :0.346203"
"釈由美子    cosine value :0.319401"
"本木雅弘    cosine value :0.291428"
"藤ヶ谷太輔    cosine value :0.289464"
"松田翔太    cosine value :0.274778"
```
  
### Validation
指定の項目に対して、CBOWの予測結果を出力する。  
1. DataID.csvより、確認したい項目のIDを4つ記憶する。(例:堺雅人=170, 上戸彩=346, None=152, 21=7)  
2. 以下を実行  
`$ python drama2vec.py --mode=validation --valid_list ID_NUMBERx4`
  
__例:「半沢直樹」の視聴率(堺雅人、上戸彩、None(主題歌)、21(放送時間帯))の場合__  
`$ python drama2vec.py --mode=validation --valid_list 170 346 152 7`  

Output  
```
--evaluating as validation--
input
[
    "堺雅人", 
    "上戸彩", 
    "None", 
    21
]
prediction
"over25   confidence :0.988974"
"10~15   confidence :0.00940623"
"木村拓哉   confidence :0.00151455"
```

__例:ムロツヨシで21時台に視聴率20~25%を出すためには、どの俳優とペアを組ませるべきか__  
`$ python drama2vec.py --mode=validation --valid_list 129 130 152 7`  

Output  
```
--evaluating as validation--
input
[
    "ムロツヨシ", 
    "20~25", 
    "None", 
    21
]
prediction
"木村拓哉   confidence :0.953642"
"水谷豊   confidence :0.0433904"
"otherCast   confidence :0.00192077"
```
やっぱり木村拓哉らしい  
  
