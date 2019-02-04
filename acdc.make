CC = python3.7

# CFLAGS = -O
# DEBUG = --debug
EPC = 200
# EPC = 5

G_RGX = (patient\d+_\d+)_\d+
# G_RGX = (\d+_patient\d+_\d+)_\d+
NET = ENet
# NET = ResidualUNet
# NET = Dummy
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]

PATHAKS = results/midl/pathak_loose results/midl/pathak_595 \
	results/midl/pathak_precise_upper results/midl/pathak_precise
HYBRIDS = results/midl/hybrid5_595 results/midl/hybrid10_595 results/midl/hybrid25_595 \
		  results/midl/fs5		   results/midl/fs10		 results/midl/fs25
TRN = results/midl/fs results/midl/partial \
	results/midl/size_595 results/midl/loose \
	results/midl/presize results/midl/presize_upper \
	results/midl/3d_sizeloss results/midl/3d_sizeloss_random \
	$(HYBRIDS)


GRAPH = results/midl/val_dice.png results/midl/tra_dice.png \
		results/midl/tra_loss.png \
		results/midl/val_batch_dice.png
HIST =  results/midl/val_dice_hist.png results/midl/tra_loss_hist.png \
		results/midl/val_batch_dice_hist.png
BOXPLOT = results/midl/val_batch_dice_boxplot.png results/midl/val_dice_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-midl.tar.gz

all: pack

plot: $(PLT)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN) $(PATHAKS) $(HYBRIDS)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available


# Data generation
data/acdc: data/acdc.lineage data/acdc.zip
	md5sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	rm $@_tmp/training/*/*_4d.nii.gz  # space optimization
	mv $@_tmp $@

data/MIDL: OPT = --seed=0
data/MIDL: data/acdc
	rm -rf $@_tmp $@
	$(CC) $(CFLAGS) slice_acdc.py --source_dir="data/acdc/training" --dest_dir=$@_tmp $(OPT)
	$(CC) remap_values.py $@_tmp/train/gt "{3: 3, 2: 0, 1: 0, 0: 0}"
	$(CC) remap_values.py $@_tmp/val/gt "{3: 3, 2: 0, 1: 0, 0: 0}"
	mv $@_tmp $@


# That one REALLY need s binary problem
data/MIDL-pathak: OPT = --seed=0
data/MIDL-pathak: data/acdc
	rm -rf $@_tmp $@
	$(CC) $(CFLAGS) slice_acdc.py --source_dir="data/acdc/training" --dest_dir=$@_tmp $(OPT)
	$(CC) remap_values.py $@_tmp/train/gt "{3: 1, 2: 0, 1: 0, 0: 0}"
	$(CC) remap_values.py $@_tmp/val/gt "{3: 1, 2: 0, 1: 0, 0: 0}"
	mv $@_tmp $@

data/MIDL/train/gt data/MIDL/val/gt: data/MIDL
data/MIDL-pathak/train/gt data/MIDL-pathak/val/gt: data/MIDL-pathak

weaks = data/MIDL/train/erosion data/MIDL/val/erosion \
		data/MIDL/train/random data/MIDL/val/random \
		data/MIDL/train/centroid data/MIDL/val/centroid

weak: $(weaks)

data/MIDL/train/erosion data/MIDL/val/erosion: OPT = --seed=0 --strategy=erosion_strat
data/MIDL/train/random data/MIDL/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat
data/MIDL/train/centroid data/MIDL/val/centroid: OPT = --seed=0 --width=4 --r=0 --strategy=centroid_strat
$(weaks): data/MIDL
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --base_folder=$(@D) --save_subfolder=$(@F)_tmp --selected_class 3 --filling 3 $(OPT)
	mv $@_tmp $@

data/MIDL-pathak/train/erosion data/MIDL-pathak/val/erosion: OPT = --seed=0 --strategy=erosion_strat
data/MIDL-pathak/train/erosion data/MIDL-pathak/val/erosion: data/MIDL-pathak
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --base_folder=$(@D) --save_subfolder=$(@F)_tmp --selected_class 1 --filling 1 $(OPT)
	mv $@_tmp $@

data/MIDL/train_5:
	cp -r data/MIDL/train $@_tmp
	for f in `ls $@_tmp` ; do \
		echo $$f ; \
		for im in `ls $@_tmp/img | cut -d '_' -f 1-2 | sort | uniq | tail -n +6` ; do \
			rm $@_tmp/$$f/$$im* ; \
		done \
	done
	mv $@_tmp $@
data/MIDL/train_10:
	cp -r data/MIDL/train $@_tmp
	for f in `ls $@_tmp` ; do \
		echo $$f ; \
		for im in `ls $@_tmp/img | cut -d '_' -f 1-2 | sort | uniq | tail -n +11` ; do \
			rm $@_tmp/$$f/$$im* ; \
		done \
	done
	mv $@_tmp $@
data/MIDL/train_25:
	cp -r data/MIDL/train $@_tmp
	for f in `ls $@_tmp` ; do \
		echo $$f ; \
		for im in `ls $@_tmp/img | cut -d '_' -f 1-2 | sort | uniq | tail -n +26` ; do \
			rm $@_tmp/$$f/$$im* ; \
		done \
	done
	mv $@_tmp $@

# Trainings
results/midl/size results/midl/random results/midl/erosion: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [3]}, 'TagBounds', {'values': {3: [97.9, 1722]}, 'idc': [3]}, 'soft_size', 1e-2)]"
results/midl/size_595: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [3]}, 'TagBounds', {'values': {3: [60, 2000]}, 'idc': [3]}, 'soft_size', 1e-2)]"
results/midl/upper: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [3]}, 'TagBounds', {'values': {3: [1, 1722]}, 'idc': [3]}, 'soft_size', 1e-2)]"
results/midl/upper_595: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [3]}, 'TagBounds', {'values': {3: [1, 2000]}, 'idc': [3]}, 'soft_size', 1e-2)]"
results/midl/loose: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [3]}, 'TagBounds', {'values': {3: [1, 65000]}, 'idc': [3]}, 'soft_size', 1e-2)]"
results/midl/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)]"
results/midl/partial: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1)]"
results/midl/presize_aug: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [3]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 0.05)]" \
	--batch_size=4
results/midl/presize: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [3]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 0.05)]"
results/midl/presize_upper: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [3]}, 'PreciseUpper', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 0.05)]"
results/midl/3d_sizeloss results/midl/3d_sizeloss_random: OPT = --losses="[('CrossEntropy', {'idc': [3]}, None, None, None, 1), \
	('BatchNaivePenalty', {'idc': [3], 'margin': 0}, 'PreciseBounds', {'margin': 0, 'mode': 'percentage'}, 'soft_size', 1e-2)]" \
	--group_train

results/midl/pathak_precise: OPT = --losses="[('PathakLoss', {'mask_idc': [1], 'idc': [0, 1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1)]" \
	--n_class=2 --metric_axis=1
results/midl/pathak_precise_upper: OPT = --losses="[('PathakUpper', {'mask_idc': [1], 'idc': [0, 1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1)]" \
	--n_class=2 --metric_axis=1
results/midl/pathak_loose: OPT = --losses="[('PathakUpper', {'mask_idc': [1], 'idc': [0, 1]}, 'TagBounds', {'values': {1: [1, 65000]}, 'idc': [1]}, 'soft_size', 1)]" \
	--n_class=2 --metric_axis=1
results/midl/pathak_595: OPT = --losses="[('PathakLoss', {'mask_idc': [1], 'idc': [0, 1]}, 'TagBounds', {'values': {1: [60, 2000]}, 'idc': [1]}, 'soft_size', 1)]" \
	--n_class=2 --metric_axis=1

results/midl/hybrid5_595: OPT = --losses="[[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)], \
		[('CrossEntropy', {'idc': [3]}, None, None, None, 1), \
	     ('NaivePenalty', {'idc': [3]}, 'TagBounds', {'values': {3: [60, 2000]}, 'idc': [3]}, 'soft_size', 1e-2)]]" \
	    --training_folders train_5 train
results/midl/hybrid10_595: OPT = --losses="[[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)], \
		[('CrossEntropy', {'idc': [3]}, None, None, None, 1), \
	     ('NaivePenalty', {'idc': [3]}, 'TagBounds', {'values': {3: [60, 2000]}, 'idc': [3]}, 'soft_size', 1e-2)]]" \
	    --training_folders train_10 train
results/midl/hybrid25_595: OPT = --losses="[[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)], \
		[('CrossEntropy', {'idc': [3]}, None, None, None, 1), \
	     ('NaivePenalty', {'idc': [3]}, 'TagBounds', {'values': {3: [60, 2000]}, 'idc': [3]}, 'soft_size', 1e-2)]]" \
	    --training_folders train_25 train
results/midl/fs5: OPT = --losses="[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)]" \
		--training_folders train_5
results/midl/fs10: OPT = --losses="[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)]" \
		--training_folders train_10
results/midl/fs25: OPT = --losses="[('CrossEntropy', {'idc': [0, 3]}, None, None, None, 1)]" \
		--training_folders train_25



# Data dependencies
results/midl/fs: data/MIDL/train/gt data/MIDL/val/gt
results/midl/fs: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/midl/partial: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/partial: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True)]"

results/midl/size: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/size: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/midl/size_595: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/size_595: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/midl/upper: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/upper: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/midl/upper_595: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/upper_595: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/midl/loose: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/loose: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/midl/presize: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/presize: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/midl/presize_upper: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/presize_upper: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

pathaks = results/midl/pathak_precise results/midl/pathak_precise_upper results/midl/pathak_loose results/midl/pathak_595
$(pathaks): data/MIDL-pathak/train/erosion data/MIDL-pathak/val/erosion
$(pathaks): DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True)]"

results/midl/random: data/MIDL/train/random data/MIDL/val/random
results/midl/random: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/midl/erosion: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/erosion: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/midl/3d_sizeloss: data/MIDL/train/erosion data/MIDL/val/erosion
results/midl/3d_sizeloss: DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/midl/3d_sizeloss_random: data/MIDL/train/random data/MIDL/val/random
results/midl/3d_sizeloss_random: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/midl/hybrid5_595: data/MIDL/train/erosion data/MIDL/val/erosion data/MIDL/train_5
results/midl/hybrid5_595: DATA = --folders="[$(B_DATA)+[('gt', gt_transform, True)], \
										     $(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]]"
results/midl/hybrid10_595: data/MIDL/train/erosion data/MIDL/val/erosion data/MIDL/train_10
results/midl/hybrid10_595: DATA = --folders="[$(B_DATA)+[('gt', gt_transform, True)], \
										      $(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]]"
results/midl/hybrid25_595: data/MIDL/train/erosion data/MIDL/val/erosion data/MIDL/train_25
results/midl/hybrid25_595: DATA = --folders="[$(B_DATA)+[('gt', gt_transform, True)], \
										      $(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]]"

data/MIDL/train_5/gt: data/MIDL/train_5
results/midl/fs5: data/MIDL/train_5/gt data/MIDL/val/gt
results/midl/fs5: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

data/MIDL/train_10/gt: data/MIDL/train_10
results/midl/fs10: data/MIDL/train_10/gt data/MIDL/val/gt
results/midl/fs10: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

data/MIDL/train_25/gt: data/MIDL/train_25
results/midl/fs25: data/MIDL/train_25/gt data/MIDL/val/gt
results/midl/fs25: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/midl/%:
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=1 --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=4 --metric_axis=3 \
		--grp_regex="$(G_RGX)" --network=$(NET) --in_memory $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@

# Need to fix pathak housefire if we want to plot anything
results/midl/pathak_%_norm: results/midl/pathak_%
	rm -rf $@ $@_tmp
	cp -r $< $@_tmp
	$(CC) remap_values.py $@_tmp/best_epoch/val "{1: 3, 0: 0}"
	for metric in `ls $@_tmp/*dice.npy` ; do \
		$(CC) remap_metric.py $$metric "{0: 0, 1: 3}" ; \
	done
	mv $@_tmp $@



# Plotting
results/midl/val_batch_dice.png results/midl/val_dice.png results/midl/val_haussdorf.png results/midl/tra_dice.png : COLS = 3
results/midl/tra_loss.png: COLS = 0
results/midl/val_dice.png results/midl/tra_loss.png results/midl/val_batch_dice.png results/midl/val_haussdorf.png: plot.py $(TRN)
results/midl/tra_dice.png : plot.py $(TRN)
results/midl/val_haussdorf.png: OPT = --ylim 0 7 --min

results/midl/val_batch_dice_hist.png results/midl/val_dice_hist.png: COLS = 3
results/midl/tra_loss_hist.png: COLS = 0
results/midl/val_dice_hist.png results/midl/tra_loss_hist.png results/midl/val_batch_dice_hist.png: hist.py $(TRN)

results/midl/val_batch_dice_boxplot.png results/midl/val_dice_boxplot.png: COLS = 3
results/midl/val_batch_dice_boxplot.png results/midl/val_dice_boxplot.png: moustache.py $(TRN)

results/midl/%.png:
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT) $(DEBUG)

metrics: $(TRN)
	$(CC) $(CFLAGS) metrics.py --num_classes=4 --grp_regex="$(G_RGX)" --gt_folder data/MIDL/val/gt \
		--pred_folders $(addsuffix /best_epoch/val, $^) $(DEBUG)

# Viewing
view: $(TRN)
	viewer -n 3 --img_source data/MIDL/val/img data/MIDL/val/gt $(addsuffix /best_epoch/val, $^) --crop 30 \
		--display_names gt $(notdir $^) --remap "{1: 0, 2: 0}" $(DEBUG)

report: $(TRN)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_batch_dice val_dice --axises 3