# Globally and Locally Consistent Image Completion

This is a Tensorflow2 Keras implementation of ["Globally and Locally Consistent Image Completion"](http://hi.cs.waseda.ac.jp/%7Eiizuka/projects/completion/data/completion_sig2017.pdf).

## Install python libraries

```
pip install pipenv
cd image_completion_keras
pipenv install
```

## Prepare dataset

Please download appropriate image dataset and put it under the "data" folder and "test" folder.
For example, please put the images as follows.

```
data
└── place365
    ├── Places365_00000001.jpg
    ├── Places365_00000002.jpg
    ├── Places365_00000003.jpg
    ├── Places365_00000004.jpg
    ├── Places365_00000005.jpg
    ├── Places365_00000006.jpg
    ├── Places365_00000007.jpg
    ├── Places365_00000008.jpg
    ...
test
└── place365_test
    ├── Places365_test_00000001.jpg
    ├── Places365_test_00000002.jpg
    ├── Places365_test_00000003.jpg
    ├── Places365_test_00000004.jpg
    ├── Places365_test_00000005.jpg
    ├── Places365_test_00000006.jpg
    ├── Places365_test_00000007.jpg
    ├── Places365_test_00000008.jpg
    ...
```

## Train

```
pipenv shell
python train.py
```

## Test

```
pipenv shell
python test.py
```

## Result

![result](output/result.png)
