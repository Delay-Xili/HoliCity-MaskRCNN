# HoliCity-MaskRCNN

Training the HoliCity V1 through MaskRCNN (Detectron2)



### Installing


[See INSTALL.md.](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)



## Dataset

Download the HoliCityV1 dataset from [holicity.io](https://people.eecs.berkeley.edu/~zyc/holicity/), 
which including [split-v1](https://drive.google.com/file/d/1Uypum27IGCxIn4JQkgWJoKhhEmh3x_WS/view), [image](https://drive.google.com/file/d/11-u2uUzBJeKDT3sGz0K-wHJtLXY4NzJD/view), [plane](https://drive.google.com/file/d/1Q3bAl66US_ZfJ_QcaSNoJ6AqnfNBqyd4/view) . 
Then, unzip them in the folder dataset/ and reorganized as follows:
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
            xxxx.txt
            ...
```
**The clean-filelist.txt existed in dataset/, 
pls put it in dataset/split/v1, you can't build any coco format json files without it**. 
The sub folder of plane/ should be same as image/

### Traning

The default batch size assumes your have a graphics card with 8GB video memory, e.g., GTX 1080Ti or RTX 2080Ti. 
You may reduce the batch size if you have less video memory.

```
CUDA_VISIBLE_DEIVCES=0 python main.py -s train -m HoliCityV1
```
It will build the train (HoliCityV1_train_coco_format.json) and valid (HoliCityV1_valid_coco_format.json) json file 
in folder data/HoliCityV1_v1/ first, it will cost 1.5 and 0.5 hours respectively.

### Detect planes for Your Own Images
To test the pretrained MaskRCNN above on your own images, you need change the HoliCity/init.py
```
self.ckpt = /the/checkpointfile/you/trained/

def predict(self):
    self.img_dirs_list = [the paths list of your own images]
``` 
 
and execute
```
CUDA_VISIBLE_DEIVCES=0 python main.py -s predict -m HoliCityV1
```


## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds


## Authors

* [**Xili Dai**](https://github.com/Delay-Xili) - *Initial work*
* [Yichao zhou](https://github.com/zhou13)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Yichao zhou](https://github.com/zhou13)


