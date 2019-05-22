# How to use
## Prediction
This project, I improve model which was trained, so you can download it, and use it to prediction!
* this project just support every sentences with 45 char length
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
        
### Parameter
| name | type | detail |
|--------------------|------|-------------|
gpu_no | int | which gpu will be use to init bert ner graph
log_dir | str | log dir 
verbose | bool| whether show tensorflow log
bert_sim_model | str| bert sim model path

## Train
### Code
In this project, I just use bert pre model to fine tuning, so I just use their original code. I try to create new one, but 
the new one just same as the original code, so I given up.
### Dataset
Because of my domain work, my work is based on judicial examination education, so I didn't use common dataset, my dataset were 
labeled by manual work, it include 80000+, 50000+ are similar, 30000+ are dissimilar, this is my dataset [link](https://pan.baidu.com/s/11qaoz6Lgd8oxkESLpKdk-A)
pwd ptje
### Suggest:
In original code, they just got the model pool output, I think there may be other ways to increase the accuracy, I tried some ways to increase the accuracy, but I found one,
just concat the [CLS] embedding of the fourth from bottom to tailender in encoder output list, if you want to use my way, just do like this。
* Delete the following code

````python
output_layer = model.get_pooled_output()
````

* Use the following code, it can **increase the accuracy 1%**.
````python
output_layer = [tf.squeeze(model.all_encoder_layers[i][:, 0:1, :], axis=1) for i in range(-4, 0, 1)]
````