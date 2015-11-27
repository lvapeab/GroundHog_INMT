dict(
path = "/home/lvapeab/smt/software/GroundHog/tutorials/DATA/ue.npz",
dictionary = "/home/lvapeab/smt/software/GroundHog/tutorials/DATA/ue_dict.npz",
chunks = 'words',
n_in = 4745,
n_out = 4745,

trainFreq = 500,
hookFreq = 1000,
validFreq = 1000,


inp_nhids = '[400]',
dout_nhid = '400',
nhids = '[400]',

seqlen = 40,
join = 'concat',

prefix = '/home/lvapeab/smt/software/GroundHog/tutorials/models/ue/en_400_400'
)
