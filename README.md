# How to use
* this project just support every sentence with 45 char length
* [download](https://pan.baidu.com/s/1CbKiY8GBGaF2dnMioLDU5Q) model file, pwd: vv1k
* just use like this 
    * first
        ````python
        bs = BertSim(gpu_no=0, log_dir='log/', bert_sim_dir='bert_sim_model\\', verbose=True)
    * second
        > similarity sentences
        ````python
        text_a = '技术侦查措施只能在立案后采取'
        text_b = '未立案不可以进行技术侦查'
        bs.predict([[text_a, text_b]])
        ````
        > you will get result like this:
        [[0.00942544 0.99057454]]
        
        > not similarity sentence
        ```python
        text_a = '华为还准备起诉美国政府'
        text_b = '飞机出现后货舱火警信息'
        bs.predict([[text_a, text_b]])
        ```
        > you will get result like this:
        [[0.98687243 0.01312758]]
        
# Parameter
| name | type | detail |
|--------------------|------|-------------|
gpu_no | int | which gpu will be use to init bert ner graph
log_dir | str | log dir 
verbose | bool| whether show tensorflow log
bert_sim_model | str| bert sim model path