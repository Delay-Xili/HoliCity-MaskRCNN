# HoliCity-MaskRCNN


Training the HoliCity V1 through MaskRCNN (Detectron2).
<img src="https://people.eecs.berkeley.edu/~zyc/holicity/images/surface-segmentations-pazo2.jpg">



### Installing


[See INSTALL.md.](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)



## Dataset

Downloading the **HoliCityV1** dataset from the homepage [holicity.io](https://people.eecs.berkeley.edu/~zyc/holicity/), 
which includes [split-v1](https://drive.google.com/file/d/1Uypum27IGCxIn4JQkgWJoKhhEmh3x_WS/view), [image](https://drive.google.com/file/d/11-u2uUzBJeKDT3sGz0K-wHJtLXY4NzJD/view), [plane](https://drive.google.com/file/d/1Q3bAl66US_ZfJ_QcaSNoJ6AqnfNBqyd4/view) . 
Unzip into the folder `dataset/` and reorganized as follows: (**The clean-filelist.txt already existed in the folder `dataset/`**. )
```
dataset/
    image/
        2008-07/
        2008-09/
        ...
    plane/
        2008-07/
        2008-09/
        ...
    split/
        v1/
            clean-filelist.txt
            filelist.txt
            train-middlesplit.txt
            test-middlesplit.txt
            valid-middlesplit.txt
```


### Pre-trained Models

You can download our reference pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1GBvZ-V_Bzanwa_RFZjReTtd6cdeT0hCb?usp=sharing). 
Those models were trained with `HoliCity/init.py` for 100k iterations.

### Traning

The default batch size assumes your have a graphics card with 8GB video memory, e.g., GTX 1080Ti or RTX 2080Ti. 
You may reduce the batch size if you have less video memory.

```
CUDA_VISIBLE_DEIVCES=0 python main.py -s train -m HoliCityV1
```
It will build the train (HoliCityV1_train_coco_format.json) and valid (HoliCityV1_valid_coco_format.json) json file 
in the folder `data/HoliCityV1_v1/` first. (It will cost about 1.5 and 0.5 hours respectively.)

### Detect planes for Your Own Images
To test the pretrained MaskRCNN above on your own images, you need change the `HoliCity/init.py`
```
self.ckpt = /the/checkpointfile/you/trained/

def predict(self):
    self.img_dirs_list = [the paths list of your own images]
``` 
 
and execute
```
CUDA_VISIBLE_DEIVCES=0 python main.py -s predict -m HoliCityV1
```


## Authors

* [**Xili Dai**](https://github.com/Delay-Xili)
* [Yichao Zhou](https://github.com/zhou13)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Yichao Zhou](https://github.com/zhou13)


