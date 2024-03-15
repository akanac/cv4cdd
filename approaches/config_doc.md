Here all configuration variables from the object detection config.py are explained in more detail.

DEBUG := (bool) specifies debugging state; defaults to True.

OBJECT_DETECTION := (bool) specifies object detection state; defaults to True.

ANNOTATIONS_ONLY := (bool) specifies whether only annotations should be created during preprocessing and skips the actual preprocessing; defaults to False.

AUTOMATE_TFR_SCRIPT := (bool) specifies whether to automate TFR script, needs path of TENSORFLOW_MODELS_DIR and calls coco script; defaults to True.

VDD_PREPROCESSING := (bool) specifies which encoding to use; defaults to False.

KEEP_AXIS := (bool) specifies whether axis should be kept for VDD encoding; defaults to False.

WINDOWS_SYSTEM := (bool) specifies whether system is Windows or unix-based; defaults to True.

MINE_CONSTRAINTS := (bool) specifies whether to mine constraints with MINERful; defaults to True.

CONSTRAINTS_DIR := (str) specifies directory of already mined constraints if MINE_CONSTRAINTS is False.

N_WINDOWS := (int) specifies number of windows for WINSIM encoding; defaults to 200.

DEFAULT_DATA_DIR := (str) specifies default data output directory.

DEFAULT_LOG_DIR := (str) specifies event log directory.

TFR_RECORDS_DIR := (str) specifies directory where to save TFR files.

TENSORFLOW_MODELS_DIR := (str) specifies TensorFlow model garden directory. The TF model garden is available here.

MINERFUL_SCRIPTS_DIR := (str) specifies MINERful directory. MINERful is available here

OUTPUT_PREFIX := (str) specifies output prefix for TFR file.

DRIFT_TYPES := (list[str]) specifies drift types; defaults to ["sudden", "gradual", "incremental", "recurring"].

DISTANCE_MEASURE := (str) specifies similarity measure; defaults to "cos". Can be one of ["fro","nuc","inf","l2","cos","earth"].

COLOR := (str) specifies whether images should be in color or grayscale; defaults to "color".

RESIZE_SUDDEN_BBOX := (bool) specifies whether to resize bounding box for sudden drifts; defaults to True.

RESIZE_VALUE := (int) specifies resize value for resizing bboxes for sudden drifts; defaults to 5.

SUB_L := (int) specifies number of sublogs for VDD encoding; defaults to 100.

SLI_BY := (int) specifies sliding by value for VDD encoding; defaults to 50.

CP_ALL := (bool) specifies whether to detect all changepoints for VDD encoding; defaults to True.

FACTOR := (int) specifies training length factor; defaults to 500.

TRAIN_EXAMPLES := (int) specifies number of training examples.

EVAL_EXAMPLES := (int) specifies number of validation examples.

TRAIN_BATCH_SIZE := (int) specifies training batch size; defaults to 64.

EVAL_BATCH_SIZE := (int) specifies validation batch size; defaults to 32.

STEPS_PER_LOOP := (int) specifies training steps per iteration; defaults to TRAIN_EXAMPLES // TRAIN_BATCH_SIZE.

TRAIN_STEPS := (int) specifies total training steps; defaults to FACTOR * STEPS_PER_LOOP.

VAL_STEPS := (int) specifies validation steps per iteration; defaults to EVAL_EXAMPLES // EVAL_BATCH_SIZE.

SUMMARY_INTERVAL := (int) specifies summary interval; defaults to STEPS_PER_LOOP.

CP_INTERVAL := (int) specifies checkpoint interval; defaults to STEPS_PER_LOOP.

VAL_INTERVAL := (int) specifies validation interval; defaults to STEPS_PER_LOOP.

EVAL_THRESHOLD := (float) specifies prediction confidence threshold; defaults to 0.5.

IMAGE_SIZE := (tuple[int]) specifies image size for training; defaults to (256, 256).

TARGETSIZE := (int) specifies targetsize for training images; defaults to 256.

N_CLASSES := (int) specifies number of classes; defaults to len(DRIFT_TYPES).

SCALE_MAX := (float) specifies scale augmentation maximum; defaults to 2.0.

SCALE_MIN := (float) specifies scale augmentation minimum; defaults to 0.1.

LR_DECAY := (bool) specifies whether learning rate should decay during training; defaults to True.

LR_INITIAL := (float) specifies initial learning rate; defaults to 1e-3.

LR_WARMUP := (float) specifies warmup learning rate; defaults to 2.5e-4.

LR_WARMUP_STEPS := (int) specifies number of learning rate warmup steps; defaults to 0.1 * TRAIN_STEPS.

BEST_CP_METRIC := (str) specifies best changepoint metric; defaults to "AP".

BEST_CP_METRIC_COMP := (str) specifies comparison for cp metric; defaults to "higher".

OPTIMIZER_TYPE := (str) specifies optimizer; defaults to "sgd".

LR_TYPE := (str) specifies learning rate decay type; defaults to "stepwise".

SGD_MOMENTUM := (float) specifies momentum for SGD; defaults to 0.9.

SGD_CLIPNORM := (float) specifies clipnorm for SGD; defaults to 10.0.

ADAM_BETA_1 := (float) specifies beta_1 for adam; defaults to 0.9.

ADAM_BETA_2 := (float) specifies beta_2 for adam; defaults to 0.999.

STEPWISE_BOUNDARIES := (list[float]) specifies stepwise boundaries for decay; defaults to [0.95 * TRAIN_STEPS,0.98 * TRAIN_STEPS].

STEPWISE_VALUES := (list[float]) specifies stepwise values; defaults to [0.32 * TRAIN_BATCH_SIZE / 256.0, 0.032 * TRAIN_BATCH_SIZE / 256.0, 0.0032 * TRAIN_BATCH_SIZE / 256.0].

MODEL_SELECTION := (str) specifies model; defaults to "retinanet_spinenet_coco".

SPINENET_ID := (int) specifies which spine net to use; defaults to "143".

TRAIN_DATA_DIR := (str) specifies training data file in TFR format.

EVAL_DATA_DIR := (str) specifies validation data file in TFR format.

MODEL_PATH := (str) specifies logging directory.

DEFAULT_OUTPUT_DIR := (str) specifies default output directory for results.

TRAINED_MODEL_PATH := (str) specifies path of pretrained/trained model.

TEST_IMAGE_DATA_DIR := (str) specifies directory where evaluation images are saved.

RELATIVE_LAG := (list[float]) specifies relative lag values for evaluation; defaults to [0.01, 0.025, 0.05, 0.1, 0.15, 0.2].

EVAL_MODE := (str) specifies evaluation mode, for naming purposes.

PRODRIFT_DIR := (str) specifies directory where ProDrift is stored.

VDD_DIR := (str) specifies directory where VDD is stored.