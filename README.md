<a name="logo"/>
<div align="center">
<a href="https://github.com/AlicanAKCA/pixera" target="_blank">
<img src="img/logo.jpg" alt="" width="400" height="200"></img>
</a>
</div>

# pixera
Pixera is a new dimension for GAN (Generative Adversarial Network). Provides to create 'pixed' images for Pixel-Art. The point you need to pay attention to is that the computer will do this job!

**Please contact me via my accounts if you wish to add new method(s).**

## GAN and pixera

The pixera provides to change image type as you can see below. Method that used below is one of the ways that can be used.

<a name="logo"/>
<div align="center">
<a href="https://github.com/AlicanAKCA/pixera" target="_blank">
<img src="img/method_1.png" alt="" width="1024" height="252"></img>
</a>
</div>

Dataset that extracted to directory with pixera prepares for [GAN method](https://arxiv.org/abs/1406.2661v1). After the this point, you can train model that include 'pixed' images with GAN method.



## Source Code Organization

| Directory         | Contents                                                           |
| -                 | -                                                                  |
| `src/`           | Includes main GAN method. |
| `dataset/`         | Dataset directory |
| `examples/`            | Will have added eternal examples, Fasten your belts!  |
| `methods/`            | Directory which includes method(s).  |
| `results/`            | Will have included consequences of training that generated model and images.   |

## Initialization

This way suggest for developers who want to create new method. Python notebooks in the examples directory can be used by developers who want to design their own trained model.

```python
 !git clone https://github.com/AlicanAKCA/pixera
```

Images that you will used, should be storaged in `/dataset/original/`. After that, run `main.py`! Please don't forget to check out pixel size you want in this file.

```python
 !python main.py
```

## Example(s)

#### Anime Faces

<a name="logo"/>
<div align="center">
<a href="https://github.com/AlicanAKCA/pixera" target="_blank">
<img src="img/example_1.jpg" alt="" width="640" height="250"></img>
</a>
</div>

* Run it on Kaggle for minimal errors. Have fun! [Kaggle Notebook](https://www.kaggle.com/alicanakca/gan-example)
