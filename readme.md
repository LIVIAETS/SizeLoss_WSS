# Constrained-CNN losses for weakly supervised segmentation
Code of our submission at [MIDL 2018](https://openreview.net/forum?id=BkIBHb2sG) and its [Medical Image Analysis](https://doi.org/10.1016/j.media.2019.02.009) journal extension. Video of the MIDL talk is available: https://www.youtube.com/watch?v=2-0Ey5-If7o

## Requirements
Non-exhaustive list:
* python3.6+
* Pytorch 1.0
* nibabel
* Scipy
* NumPy
* Matplotlib
* Scikit-image
* zsh

## Usage
Instruction to download the data are contained in the lineage files [acdc.lineage](data/acdc.lineage), [zenodo_spine](data/zenodo_spine.lineage) and [prostate.lineage](data/prostate.lineage). They are just text files containing the md5sum (or sha256sum) of the original zip.

Once the zip is in place, everything should be automatic:
```
make -f acdc.make
make -f prostate.make
make -f zenodo_spine.make
```
Usually takes a little bit more than a day per makefile.

This perform in the following order:
* Unpacking of the data
* Remove unwanted big files
* Normalization and slicing of the data
* Training with the different methods
* Plotting of the metrics curves
* Display of a report
* Archiving of the results in an .tar.gz stored in the `archives` folder

The main advantage of the makefile is that it will handle by itself the dependencies between the different parts. For instance, once the data has been pre-processed, it won't do it another time, even if you delete the training results. It is also a good way to avoid overwriting existing results by relaunching the exp by accident.

Of course, parts can be launched separately :
```
make -f acdc.make data/acdc # Unpack only
make -f acdc.make data/MIDL # unpack if needed, then slice the data
make -f acdc.make results/acdc/fs # train only with full supervision. Create the data if needed
make -f acdc.make results/acdc/val_dice.png # Create only this plot. Do the trainings if needed
```
The number of option for the main script is fairly dense, but the recipes in the different makefiles should give you a good idea on how to modify the training parameters and create new targets. In case of questions, feel free to contact me.

## Data scheme
### datasets
For instance
```
MIDL/
    train/
        img/
            case_10_0_0.png
            ...
        gt/
            case_10_0_0.png
            ...
        random/
            ...
        ...
    val/
        img/
            case_10_0_0.png
            ...
        gt/
            case_10_0_0.png
            ...
        random/
            ...
        ...
```
The network takes png files as an input. The gt folder contains gray-scale images of the ground-truth, where the gray-scale level are the number of the class (namely, 0 and 1). This is because I often use my [segmentation viewer](https://github.com/HKervadec/segmentation_viewer) to visualize the results, so that does not really matter. If you want to see it directly in an image viewer, you can either use the remap script, or use imagemagick:
```
mogrify -normalize data/ISLES/val/gt/*.png
```

### results
```
results/
    acdc/
        fs/
            best_epoch/
                val/
                    case_10_0_0.png
                    ...
            iter000/
                val/
            ...
        size_595/
            ...
        best.pkl # best model saved
        metrics.csv # metrics over time, csv
        best_epoch.txt # number of the best epoch
        val_dice.npy # log of all the metric over time for each image and class
        val_dice.png # Plot over time
        ...
    prostate/
        ...
archives/
    $(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-acdc.tar.gz
    $(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-prostate.tar.gz
```

## Interesting bits
The losses are defined in the [`losses.py`](losses.py) file. Explaining the remaining of the code is left as an exercise for the reader.

## Cool tricks
Remove all assertions from the code. Usually done after making sure it does not crash for one complete epoch:
```
make -f acdc.make <anything really> CFLAGS=-O
```

Use a specific python executable:
```
make -f acdc.make <super target> CC=/path/to/the/executable
```

Train for only 5 epochs, with a dummy network, and only 10 images per data loader. Useful for debugging:
```
make -f acdc.make <really> NET=Dimwit EPC=5 DEBUG=--debug
```

Rebuild everything even if already exist:
```
make -f acdc.make <a> -B
```

Only print the commands that will be run (useful to check recipes are properly defined):
```
make -f acdc.make <a> -n
```

Create a gif for the predictions over time of a specific patient:
```
cd results/acdc/fs
convert iter*/val/case_14_0_0.png case_14_0_0.gif
mogrify -normalize case_14_0_0.gif
```
