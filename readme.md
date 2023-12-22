
# Cifar10

### Paper "Emerging properties in Vision Transformers"
A vision transformer was trained with only 10k images to simulate a shortage of labeled data.

|               |  Supervised |  Dino v1  |  Dino v2 |
|---------------|-------------|-----------|----------|
| Val. Loss     |  1.412      |  1.258    |  1.206   |
| Val. Accuracy |  58 %       |  62 %     |  63.5 %  |

As proposed in the paper, after a while of training segmentation masks appear in the
attention maps (in some cases).
<img src="images/emerging_properties_car.png" width="500px"/>