from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.FROM_SCRATCH = True
_C.DATA_TRUNK = None

_C.OUTPUT_DIR = 'data/vidstg/checkpoints/output'
_C.DATA_DIR = 'data/vidstg'
_C.GLOVE_DIR = ''
_C.TENSORBOARD_DIR = 'data/vidstg/checkpoints/output/tensorboard'

# -----------------------------------------------------------------------------
# BERT
# -----------------------------------------------------------------------------
_C.BERT = CN()
_C.BERT.BERT_PATH = '../bert/bert-base-uncased'
_C.BERT.CORENLP_PAYH = '../corenlp/stanford-corenlp-4.5.1'
_C.BERT.MAX_LEN = 32

# -----------------------------------------------------------------------------
# STVMR
# -----------------------------------------------------------------------------
_C.STVMR = CN()
_C.STVMR.INSTANCE_TYPE = 'frame'
_C.STVMR.CLIP_LENGTH = [4, 8, 12, 16, 20]
_C.STVMR.MAX_CLIP_NUM = 20
_C.STVMR.SIGMA = 0.5
_C.STVMR.TAU = 0.1
_C.STVMR.DELTA_TEMPORAL = 10
_C.STVMR.DELTA_SPATIO = 5
_C.STVMR.LAMDA = 0.9
_C.STVMR.SENTENCE_SHAPE = 768
_C.STVMR.REGION_SHAPE = 4096
_C.STVMR.FEATURE_SHAPE = 512
_C.STVMR.LR = 1e-4
_C.STVMR.WEIGHT_DECAY = 1e-2
_C.STVMR.THRESHOLD = 0.5
_C.STVMR.IS_GAUSS_LOSS = True

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.MAX_QUERY_LEN = 26
_C.INPUT.MAX_VIDEO_LEN = 200

# The input frame number (For VidSTG)
_C.INPUT.TRAIN_SAMPLE_NUM = 12
# The input frame rate (For HC_STVG, 20s per input video)
_C.INPUT.SAMPLE_FPS = 3.2

_C.INPUT.VIDEO_SAMPLE_NUM = 50.0
# The input video resolution
_C.INPUT.RESOLUTION = 224
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# If perform multiscale training
_C.INPUT.AUG_SCALE = True
# If perform translate augumentation training
_C.INPUT.AUG_TRANSLATE = False
_C.INPUT.IMAGE_MAX_SIZE = 224
_C.INPUT.IMAGE_MIN_SIZE = 224
# Image ColorJitter
_C.INPUT.FLIP_PROB_TRAIN = 0.5
_C.INPUT.TEMP_CROP_PROB = 0.5

# -----------------------------------------------------------------------------
# Model Config
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
# _C.MODEL.WEIGHT = "data/pretrained/pretrained_resnet101_checkpoint.pth"
_C.MODEL.WEIGHT = "data/vidstg/checkpoints/output/model_096792.pth"
_C.MODEL.EMA = True
_C.MODEL.EMA_DECAY = 0.9998
_C.MODEL.QUERY_NUM = 1   # each frame a single query
_C.MODEL.DOWN_RATIO = 4

# -----------------------------------------------------------------------------
# Vision Encoder options
# -----------------------------------------------------------------------------

_C.MODEL.VISION_BACKBONE = CN()
_C.MODEL.VISION_BACKBONE.NAME = 'resnet101'  # resnet50 or resnet101
_C.MODEL.VISION_BACKBONE.POS_ENC = 'sine'  # sine, sineHW or learned
_C.MODEL.VISION_BACKBONE.DILATION = False # If true, we replace stride with dilation in the last convolutional block (DC5)
_C.MODEL.VISION_BACKBONE.FREEZE = False # If true, freeze the vision backbone parameters


# -----------------------------------------------------------------------------
# Language Encoder Config
# -----------------------------------------------------------------------------
_C.MODEL.TEXT_MODEL = CN()
_C.MODEL.TEXT_MODEL.NAME = 'roberta-base'  # "bert-base", "roberta-large"
_C.MODEL.TEXT_MODEL.FREEZE = False

# If true, use LSTM as the text encoder
_C.MODEL.USE_LSTM = False
_C.MODEL.LSTM = CN()
_C.MODEL.LSTM.NAME = 'lstm'
_C.MODEL.LSTM.HIDDEN_SIZE = 512
_C.MODEL.LSTM.BIDIRECTIONAL = True
_C.MODEL.LSTM.DROPOUT = 0
_C.MODEL.LSTM_NUM_LAYERS = 2


# -----------------------------------------------------------------------------
# STCAT Pipeline Config
# -----------------------------------------------------------------------------
_C.MODEL.STCAT = CN()
_C.MODEL.STCAT.HIDDEN = 256
_C.MODEL.STCAT.VISION_HIDDEN = 256
_C.MODEL.STCAT.QUERY_DIM = 4  # the anchor dim
_C.MODEL.STCAT.ENC_LAYERS = 6
_C.MODEL.STCAT.DEC_LAYERS = 6
_C.MODEL.STCAT.FFN_DIM = 2048
_C.MODEL.STCAT.DROPOUT = 0.1
_C.MODEL.STCAT.HEADS = 8
_C.MODEL.STCAT.USE_LEARN_TIME_EMBED = False
_C.MODEL.STCAT.USE_ACTION = True  # use the actioness head by default
_C.MODEL.STCAT.FROM_SCRATCH = True

# For 2D-Map prediction
_C.MODEL.STCAT.TEMP_PRED_LAYERS = 6
_C.MODEL.STCAT.CONV_LAYERS = 4
_C.MODEL.STCAT.TEMP_HEAD = 'attn'   # attn or conv
_C.MODEL.STCAT.KERNAL_SIZE = 9
_C.MODEL.STCAT.MAX_MAP_SIZE = 128
_C.MODEL.STCAT.POOLING_COUNTS = [15,8,8,8]
_C.MODEL.USE_DSDECOMPOSE = False


#stvgbert config
# _C.MODEL
_C.MODEL.ATTENTION_PROBS_DROPOUT_PROB = 0.1
_C.MODEL.HIDDEN_ACT = 'gelu'
_C.MODEL.HIDDEN_DROPOUT_PROB = 0.1
_C.MODEL.HIDDEN_SIZE = 256
_C.MODEL.INITIALIZER_RANGE = 0.02
_C.MODEL.INTERMEDIATE_SIZE = 3072
_C.MODEL.MAX_POSITION_EMBEDDINGS = 512
_C.MODEL.NUM_ATTENTION_HEADS = 12
_C.MODEL.NUM_HIDDEN_LAYERS = 12
_C.MODEL.TYPE_VOCAB_SIZE = 2
_C.MODEL.VOCAB_SIZE = 30522
_C.MODEL.V_FEATURE_SIZE = 2048
_C.MODEL.V_TARGET_SIZE = 1601
_C.MODEL.V_HIDDEN_SIZE = 256
_C.MODEL.V_NUM_HIDDEN_LAYERS = 6
_C.MODEL.V_NUM_ATTENTION_HEADS = 8
_C.MODEL.V_INTERMEDIATE_SIZE = 256
_C.MODEL.BI_HIDDEN_SIZE = 256
_C.MODEL.BI_NUM_ATTENTION_HEADS = 8
_C.MODEL.BI_INTERMEDIATE_SIZE = 256
_C.MODEL.BI_ATTENTION_TYPE = 1
_C.MODEL.V_ATTENTION_PROBS_DROPOUT_PROB = 0.1
_C.MODEL.V_HIDDEN_ACT = 'gelu'
_C.MODEL.V_HIDDEN_DROPOUT_PROB = 0.1
_C.MODEL.V_INITIALIZER_RANGE = 0.02
_C.MODEL.V_BIATTENTION_ID = [0, 1, 2, 3, 4, 5]
_C.MODEL.T_BIATTENTION_ID = [6, 7, 8, 9, 10, 11]
_C.MODEL.POOLING_METHOD = 'mul'
_C.MODEL.VISUALIZATION = False
# -----------------------------------------------------------------------------
# DATASET related params
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'VidSTG'
_C.DATASET.NUM_CLIP_FRAMES = 32
# The minimum gt frames in a sampled clip
_C.DATASET.MIN_GT_FRAME = 4


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SIZE_DIVISIBILITY = 0
_C.DATALOADER.ASPECT_RATIO_GROUPING = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 5
_C.SOLVER.BATCH_SIZE = 4   # The video number per GPU, should be set 1.
_C.SOLVER.SHUFFLE = True
_C.SOLVER.BASE_LR = 2e-5
_C.SOLVER.VIS_BACKBONE_LR = 1e-5
_C.SOLVER.TEXT_LR = 2e-5
_C.SOLVER.TEMP_LR = 1e-4
_C.SOLVER.OPTIMIZER = 'adamw'
_C.SOLVER.MAX_GRAD_NORM = 0.1

# loss weight hyper-parameter
_C.SOLVER.BBOX_COEF = 5
_C.SOLVER.GIOU_COEF = 2
_C.SOLVER.TEMP_COEF = 2
_C.SOLVER.ATTN_COEF = 1
_C.SOLVER.ACTIONESS_COEF = 2

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.POWER = 0.9    # For Poly LRScheduler
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500

_C.SOLVER.WARMUP_PROP = 0.01
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.SCHEDULE = CN()
# _C.SOLVER.SCHEDULE.TYPE = "linear_with_warmup"
_C.SOLVER.SCHEDULE.TYPE = "multistep_with_warmup"
_C.SOLVER.SCHEDULE.DROP_STEP = [8,12]

# the following paramters are only used for WarmupReduceLROnPlateau
_C.SOLVER.SCHEDULE.PATIENCE = 2
_C.SOLVER.SCHEDULE.THRESHOLD = 1e-4
_C.SOLVER.SCHEDULE.COOLDOWN = 1
_C.SOLVER.SCHEDULE.FACTOR = 0.5
_C.SOLVER.SCHEDULE.MAX_DECAY_STEP = 7

_C.SOLVER.PRE_VAL = False
_C.SOLVER.TO_VAL = True
_C.SOLVER.VAL_PERIOD = 2500    # every 10% training iterations completed, start a avaluation
_C.SOLVER.CHECKPOINT_PERIOD = 5000


_C.SOLVER.USE_ATTN = True  # whether to use the guided attention loss, to compare with TubeDETR
_C.SOLVER.SIGMA = 2.0  # standard deviation for the quantized gaussian law used for the kullback leibler divergence loss
_C.SOLVER.USE_AUX_LOSS = True # whether to use auxiliary decoding losses (loss at each layer)
_C.SOLVER.EOS_COEF = 0.1  # The coeff for negative sample