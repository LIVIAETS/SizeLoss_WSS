CC = python3.7

# CFLAGS = -O
# DEBUG = --debug
EPC = 200
# EPC = 5


G_RGX = (\d+_Case\d+_\d+)_\d+
NET = ResidualUNet
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]

SIZES = results/prostate/sizeloss_e results/prostate/sizeloss_r
TRN = results/prostate/fs results/prostate/partial results/prostate/presize \
	results/prostate/sizeloss_r \
	results/prostate/loose results/prostate/presize_upper
	# results/prostate/3d_sizeloss



GRAPH = results/prostate/val_dice.png results/prostate/tra_loss.png \
		results/prostate/val_batch_dice.png results/prostate/tra_dice.png
HIST =  results/prostate/val_dice_hist.png results/prostate/tra_loss_hist.png \
		results/prostate/val_batch_dice_hist.png
BOXPLOT = results/prostate/val_batch_dice_boxplot.png results/prostate/val_dice_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)


REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-prostate.tar.gz

all: pack

plot: $(PLT)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	# tar -zc -f $@ $^  # Use if pigz is not available
	tar cf - $^ | pigz > $@
	chmod -w $@


# Extraction and slicing
data/PROSTATE/train/gt data/PROSTATE/val/gt: data/PROSTATE
data/PROSTATE: data/promise
	rm -rf $@_tmp
	$(CC) $(CFLAGS) slice_promise.py --source_dir $< --dest_dir $@_tmp --n_augment=0
	mv $@_tmp $@
data/promise: data/prostate.lineage data/TrainingData_Part1.zip data/TrainingData_Part2.zip data/TrainingData_Part3.zip
	md5sum -c $<
	rm -rf $@_tmp
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	unzip -q $(word 4, $^) -d $@_tmp
	mv $@_tmp $@


# Weak labels generation
weaks = data/PROSTATE/train/centroid data/PROSTATE/val/centroid \
		data/PROSTATE/train/erosion data/PROSTATE/val/erosion \
		data/PROSTATE/train/random data/PROSTATE/val/random
weak: $(weaks)

data/PROSTATE/train/centroid data/PROSTATE/val/centroid: OPT = --seed=0 --width=4 --r=0 --strategy=centroid_strat
data/PROSTATE/train/erosion data/PROSTATE/val/erosion: OPT = --seed=0 --strategy=erosion_strat
data/PROSTATE/train/random data/PROSTATE/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat

$(weaks): data/PROSTATE
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --selected_class 1 --filling 1 --base_folder=$(@D) --save_subfolder=$(@F)_tmp $(OPT)
	mv $@_tmp $@



data/PROSTATE-Aug/train/gt data/PROSTATE-Aug/val/gt: data/PROSTATE-Aug
data/PROSTATE-Aug/train/centroid data/PROSTATE-Aug/val/centroid: data/PROSTATE-Aug
data/PROSTATE-Aug/train/erosion data/PROSTATE-Aug/val/erosion: data/PROSTATE-Aug
data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random: data/PROSTATE-Aug
data/PROSTATE-Aug: data/PROSTATE $(weaks)
	rm -rf $@ $@_tmp
	$(CC) $(CFLAGS) augment.py --n_aug 4 --root_dir $</train --dest_dir $@_tmp/train
	$(CC) $(CFLAGS) augment.py --n_aug 0 --root_dir $</val --dest_dir $@_tmp/val  # Naming scheme for consistency
	mv $@_tmp $@

data/PROSTATE-aug-tiny: data/PROSTATE-Aug
	rm -rf $@ $@_tmp
	cp -r $< $@_tmp
	for f in `ls $@_tmp/train` ; do \
		mogrify -resize '128x128!' $@_tmp/train/$$f/*.png ; \
	done
	for f in `ls $@_tmp/val` ; do \
		mogrify -resize '128x128!' $@_tmp/val/$$f/*.png ; \
	done
	mv $@_tmp $@



# Training
$(SIZES): OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [60, 9000]}, 'idc': [1]}, 'soft_size', 1e-2)]"
results/prostate/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]"
results/prostate/partial: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1)]"
results/prostate/presize: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]"
results/prostate/presize_upper: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'PreciseUpper', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]"
results/prostate/loose: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [1, 65000]}, 'idc': [1]}, 'soft_size', 1e-2)]"
results/prostate/3d_sizeloss: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1), \
	('BatchNaivePenalty', {'idc': [1], 'margin': 0}, 'PreciseBounds', {'margin': 0, 'mode': 'percentage'}, 'soft_size', 1e-2)]" \
	--group_train
results/prostate/3d_sizeloss: NET = ENet


results/prostate/fs: data/PROSTATE-Aug/train/gt data/PROSTATE-Aug/val/gt
results/prostate/fs: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/prostate/sizeloss_e results/prostate/neg: data/PROSTATE-Aug/train/erosion data/PROSTATE-Aug/val/erosion
results/prostate/sizeloss_e results/prostate/neg:  DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/prostate/partial: data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random
results/prostate/partial:  DATA = --folders="$(B_DATA)+[('random', gt_transform, True)]"

results/prostate/sizeloss_c: data/PROSTATE-Aug/train/centroid data/PROSTATE-Aug/val/centroid
results/prostate/sizeloss_c: DATA = --folders="$(B_DATA)+[('centroid', gt_transform, True), ('centroid', gt_transform, True)]"

results/prostate/sizeloss_r: data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random
results/prostate/sizeloss_r: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/prostate/presize: data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random
results/prostate/presize: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/prostate/presize_upper: data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random
results/prostate/presize_upper: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/prostate/loose: data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random
results/prostate/loose: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/prostate/3d_sizeloss: data/PROSTATE-Aug/train/random data/PROSTATE-Aug/val/random
results/prostate/3d_sizeloss: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"


$(TRN):
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=4 --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=2 --metric_axis=1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# # Inference
# INFR = results/prostate/inference/fs_sanity results/prostate/inference/size_sanity
# results/prostate/inference/fs_sanity: results/prostate/fs data/PROSTATE/val
# results/prostate/inference/size_sanity: results/prostate/sizeloss data/PROSTATE/val
# $(INFR):
# 	$(CC) inference.py --save_folder $@_tmp --model_weights $</best.pkl --data_folder $(word 2, $^)/img --num_classes 2 $(OPT)
# 	$(CC) metrics.py --pred_folder $@_tmp/iter000 --gt_folder $(word 2, $^)/gt --save_folder $@_tmp  \
# 		--grp_regex="$(G_RGX)" --num_classes=2
# 	mv $@_tmp $@


# Plotting
results/prostate/val_batch_dice.png results/prostate/val_dice.png results/prostate/tra_dice.png: COLS = 1
results/prostate/tra_loss.png: COLS = 0
results/prostate/val_dice.png results/prostate/tra_loss.png results/prostate/val_batch_dice.png: plot.py $(TRN)
results/prostate/tra_dice.png: plot.py $(TRN)

results/prostate/val_haussdorf.png: COLS = 1
results/prostate/val_haussdorf.png: OPT = --ylim 0 7 --min
results/prostate/val_haussdorf.png: plot.py $(TRN)

results/prostate/val_batch_dice_hist.png results/prostate/val_dice_hist.png: COLS = 1
results/prostate/tra_loss_hist.png: COLS = 0
results/prostate/val_dice_hist.png results/prostate/tra_loss_hist.png results/prostate/val_batch_dice_hist.png: hist.py $(TRN)

results/prostate/val_batch_dice_boxplot.png results/prostate/val_dice_boxplot.png: COLS = 1
results/prostate/val_batch_dice_boxplot.png results/prostate/val_dice_boxplot.png: moustache.py $(TRN)

$(GRAPH) $(HIST) $(BOXPLOT):
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT)

# Viewing
view: $(TRN)
	viewer -n 3 --img_source data/PROSTATE/val/img data/PROSTATE/val/gt $(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^)

report: $(TRN)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_batch_dice val_dice --axises 1