CC = python3.7

# CFLAGS = -O
# DEBUG = --debug
EPC = 200
# EPC = 5

G_RGX = (Img_\d+)_\d+
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]
# NET = ResidualUNet
NET = ENet
# NET = Dummy

TRN = results/zenodo_spine/fs results/zenodo_spine/partial \
		results/zenodo_spine/sizeloss results/zenodo_spine/loose \
		results/zenodo_spine/presize results/zenodo_spine/presize_upper \
		results/zenodo_spine/sizeloss_80_2000 results/zenodo_spine/sizeloss_80_5000 \
		results/zenodo_spine/sizeloss_80_10000 results/zenodo_spine/sizeloss_0_1100 \
		results/zenodo_spine/sizeloss_0_5000 results/zenodo_spine/sizeloss_0_10000

GRAPH = results/zenodo_spine/val_dice.png results/zenodo_spine/tra_dice.png \
		results/zenodo_spine/tra_loss.png \
		results/zenodo_spine/val_batch_dice.png
HIST =  results/zenodo_spine/val_dice_hist.png results/zenodo_spine/tra_loss_hist.png \
		results/zenodo_spine/val_batch_dice_hist.png
BOXPLOT = results/zenodo_spine/val_batch_dice_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)

# Notice that this is computed when Make starts, no at the end
# If the training spans across several days, night have a difference between the name and creatinon time
REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-zenodo_spine.tar.gz

all: pack

plot: $(PLT)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@

data/zenodo_spine: data/zenodo_spine.lineage data/wetransfer-ceacbf.zip
	sha256sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	mv $@_tmp $@

data/ZENODO_SPINE: OPT = --seed=0 --shape 256 256 --retain=5 --crop 60
data/ZENODO_SPINE: data/zenodo_spine
	rm -rf $@_tmp $@
	$(CC) $(CFLAGS) slice_zenodo_spine.py --source_dir=$< --dest_dir=$@_tmp $(OPT)
	mv $@_tmp $@

data/ZENODO_3D: OPT = --seed 0 --retain 5
data/ZENODO_3D: data/zenodo_spine
	rm -rf $@_tmp $@
	$(CC) $(CFLAGS) subvolume_zenodo.py --source_dir=$< --dest_dir=$@_tmp $(OPT)
	mv $@_tmp $@



weaks = data/ZENODO_SPINE/train/centroid data/ZENODO_SPINE/val/centroid \
		data/ZENODO_SPINE/train/erosion data/ZENODO_SPINE/val/erosion \
		data/ZENODO_SPINE/train/random data/ZENODO_SPINE/val/random \
		data/ZENODO_SPINE/train/box data/ZENODO_SPINE/val/box
weak: $(weaks)

data/ZENODO_SPINE/train/gt data/ZENODO_SPINE/val/gt: data/ZENODO_SPINE

data/ZENODO_SPINE/train/centroid data/ZENODO_SPINE/val/centroid: OPT = --seed=0 --width=4 --r=0 --strategy=centroid_strat
data/ZENODO_SPINE/train/erosion data/ZENODO_SPINE/val/erosion: OPT = --seed=0 --strategy=erosion_strat
data/ZENODO_SPINE/train/random data/ZENODO_SPINE/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat
data/ZENODO_SPINE/train/box data/ZENODO_SPINE/val/box: OPT = --seed=0 --margin=5 --strategy=box_strat --allow_bigger --allow_overflow

$(weaks): data/ZENODO_SPINE
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --base_folder=$(@D) --save_subfolder=$(@F)_tmp --selected_class 1 --filling 1 $(OPT)
	mv $@_tmp $@


# Trainings
results/zenodo_spine/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/zenodo_spine/partial: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1)]"

results/zenodo_spine/sizeloss: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [80, 1100]}, 'idc': [1]}, 'soft_size', 1e-2)]"
results/zenodo_spine/presize: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1), \
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]"
results/zenodo_spine/presize_upper: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'PreciseUpper', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]"
results/zenodo_spine/loose: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [1, 65550]}, 'idc': [1]}, 'soft_size', 1e-2)]"

results/zenodo_spine/sizeloss_80_2000: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [80, 2000]}, 'idc': [1]}, 'soft_size', 1e-2)]"
results/zenodo_spine/sizeloss_80_5000: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [80, 5000]}, 'idc': [1]}, 'soft_size', 1e-2)]"
results/zenodo_spine/sizeloss_80_10000: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [80, 10000]}, 'idc': [1]}, 'soft_size', 1e-2)]"
results/zenodo_spine/sizeloss_0_1100: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [0, 1100]}, 'idc': [1]}, 'soft_size', 1e-2)]"
results/zenodo_spine/sizeloss_0_5000: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [0, 5000]}, 'idc': [1]}, 'soft_size', 1e-2)]"
results/zenodo_spine/sizeloss_0_10000: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [0, 10000]}, 'idc': [1]}, 'soft_size', 1e-2)]"

# Not augmented ones
results/zenodo_spine/fs results/zenodo_spine/partial: data/ZENODO_SPINE/train/gt data/ZENODO_SPINE/val/gt
results/zenodo_spine/fs results/zenodo_spine/partial: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/zenodo_spine/sizeloss results/zenodo_spine/loose: data/ZENODO_SPINE/train/random data/ZENODO_SPINE/val/random
results/zenodo_spine/sizeloss results/zenodo_spine/loose: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/zenodo_spine/presize: data/ZENODO_SPINE/train/random data/ZENODO_SPINE/val/random
results/zenodo_spine/presize: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/zenodo_spine/presize_upper: data/ZENODO_SPINE/train/random data/ZENODO_SPINE/val/random
results/zenodo_spine/presize_upper: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/zenodo_spine/sizeloss_80_2000 results/zenodo_spine/sizeloss_80_5000 results/zenodo_spine/sizeloss_80_10000: data/ZENODO_SPINE/train/random data/ZENODO_SPINE/val/random
results/zenodo_spine/sizeloss_80_2000 results/zenodo_spine/sizeloss_80_5000 results/zenodo_spine/sizeloss_80_10000: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/zenodo_spine/sizeloss_0_1100 results/zenodo_spine/sizeloss_0_5000 results/zenodo_spine/sizeloss_0_10000: data/ZENODO_SPINE/train/random data/ZENODO_SPINE/val/random
results/zenodo_spine/sizeloss_0_1100 results/zenodo_spine/sizeloss_0_5000 results/zenodo_spine/sizeloss_0_10000: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

# Now this thing is convenient, but then need to define the recipes for the png AFTER, otherwise will overwrite it
results/zenodo_spine/%:
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=20 --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=2 --metric_axis 1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Plotting
results/zenodo_spine/val_batch_dice.png results/zenodo_spine/val_dice.png results/zenodo_spine/tra_dice.png : COLS = 1
results/zenodo_spine/tra_loss.png: COLS = 0
results/zenodo_spine/val_dice.png results/zenodo_spine/tra_loss.png results/zenodo_spine/val_batch_dice.png: plot.py $(TRN)
results/zenodo_spine/tra_dice.png : plot.py $(TRN)

results/zenodo_spine/val_batch_dice_hist.png results/zenodo_spine/val_dice_hist.png: COLS = 1
results/zenodo_spine/tra_loss_hist.png: COLS = 0
results/zenodo_spine/val_dice_hist.png results/zenodo_spine/tra_loss_hist.png results/zenodo_spine/val_batch_dice_hist.png: hist.py $(TRN)

results/zenodo_spine/val_batch_dice_boxplot.png: COLS = 1
results/zenodo_spine/val_batch_dice_boxplot.png: moustache.py $(TRN)

results/zenodo_spine/%.png:
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT)


# Viewing
view: $(TRN)
	viewer -n 3 --img_source data/ZENODO_SPINE/val/img data/ZENODO_SPINE/val/gt $(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^) --crop 60

report: $(TRN)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_batch_dice val_dice --axises 1

metrics: $(TRN)
	$(CC) $(CFLAGS) metrics.py --num_classes=2 --grp_regex="$(G_RGX)" --gt_folder data/ZENODO_SPINE/val/gt \
		--pred_folders $(addsuffix /best_epoch/val, $^) $(DEBUG)