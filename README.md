# Collaborative Learning of Bidirectional Decoders for Unsupervised Text Style Transfer
This is the PyTorch implementation of the EMNLP2021 paper "Collaborative Learning of Bidirectional Decoders for Unsupervised Text Style Transfer"[[pdf](https://aclanthology.org/2021.emnlp-main.729.pdf)].

### Overview
- `src/` contains the implementations of the proposed CBD method
```
src
|---train_classifier.py # train the cnn classifiers used in training/evaluation
|---train_lm.py # train the language model used in evaluation
|---train_st.py # train the proposed style transfer model
|---evaluator.py # evaluate the predictions of the style transfer model

```
- `outputs/` contains the predictions of our CBD model
```
outputs
|---GYAFC
|---|---CBD.0 # formal->informal outputs of CBD model
|---|---CBD.1 # informal->formal outputs of CBD model
|---|---test.0 # formal input
|---|---test.1 # informal input
|---yelp
|---|---CBD.0 # negative->positive outputs of CBD model
|---|---CBD.1 # positive->negative outputs of CBD model
|---|---test.0 # negative input
|---|---test.1 # positive input
```

### Dependencies
```
python == 3.7
pytorch == 1.3.1
```

### Citation
If you find our paper and code useful, please cite:
```
@inproceedings{ma-etal-2021-collaborative,
    title = "Collaborative Learning of Bidirectional Decoders for Unsupervised Text Style Transfer",
    author = "Ma, Yun and Chen, Yangbin and Mao, Xudong and Li, Qing",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "9250--9266",
}

```


