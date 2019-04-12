# An Improved End-To-End Memory Network For QA Tasks in PyTorch

This repo is the PyTorch implementation of An Improved End-To-End Memory Network model based by [End-To-End Memory Networks in PyTorch](https://github.com/shihan9/MemN2N)




## Requirements

+ [PyTorch](http://pytorch.org/) ==0.4.1
+ [torchtext](https://github.com/pytorch/text) ==0.3.1
+ [click](http://click.pocoo.org/5/) ==6.7

## Dataset

The dataset is bAbI 20 QA tasks (v1.2) from [facebook research](https://research.fb.com/), which you can find it [here](https://research.fb.com/downloads/babi/).

## Benchmarks

| Task                     | MemN2N| MemN2N(Local-Attention)| 	MemN2N(GLU)|MemN2N-GL  |
| ------------------------ | --------- | -------- | -------------- | -------------- |
|   1: 1 supporting fact     |0.0	|0.0  |	 0.0	     |0.0   |  
|   2: 2 supporting facts    | 78.6|  	69.9  |  	75.8 |	66.2| 
|   3: 3 supporting facts    | 71.7|  	74.8  |  	76.6 | 	71.5| 
|   4: 2 argument relations  | 0.0 |	0.0  |	 0.0	 |0.0   |  
|   5: 3 argument relation   | 9.5 | 	11.2  |  	3.2  |	0.9 | 
|   6: yes/no questions      | 50.0|  	50.0  |  	25.2 | 	48.2|  
|   7: counting              | 50.3|  	11.5  |  	16.1 | 	10.9| 
|   8: lists/sets            | 8.7 | 	7.3   |	 6.0 	 |5.6   |
|   9: simple negation       | 12.3|  	35.2  |  	13.1 |	4.4 |
|   10:indefinite knowledg   | 11.3|  	2.6  |	 3.9 	 |13.9  |
|   11: basic coreference    | 15.9|  	18.0  |  	40.9 |	9.0 |
|   12: conjunction          |0.0	|0.0  |	 2.8 	 |0.0 |      |
|   13: compound coreference  | 46.1|  	21.3  |  	18.8 | 	1.4 |
|   14: time reasoning        | 6.9 | 	5.8   |	 4.4	 |10.3  |
|   15: basic deduction       | 43.7|  	75.8  |  	2.4  |	0.0 | 
|   16: basic induction       | 53.3|  	52.5  |  	57.8 | 	53.1|  
|   17: positional reasoning  | 46.4|  	50.8  |  	48.6 |	46.2| 
|   18: size reasoning        | 9.7 | 	7.4   |	 13.6     |	12.9|  
|   19: path finding          | 89.1|  	89.3  |  	14.5 | 	24.7|  
|   20: agent’s motivation   | 0.6 |	0.0   |	 0.0	 |1.7   |



## Usage

To train by default setting:

```shell
python cli.py
```



