dict(
path = "/home/lvapeab/smt/software/GroundHog/tutorials/DATA/miniue/fda/miniue.npz",
dictionary = "/home/lvapeab/smt/software/GroundHog/tutorials/DATA/miniue/fda/miniue_dict.npz",
chunks = 'words',
n_in = 10001,
n_out = 10001,

trainFreq = 500,
hookFreq = 5000,
validFreq = 3000,

join = 'maxPooling',

inp_nhids = '[200]',
dout_nhid = '200',
nhids = '[200]',

seqlen = 20,

prefix = '/home/lvapeab/smt/software/GroundHog/tutorials/models/ue/miniue_en_200_200'
)
