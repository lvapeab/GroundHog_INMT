from encdec import RNNEncoderDecoder
from encdec import create_padded_batch
from encdec import get_batch_iterator
from encdec import parse_input
#from nmt.online.kevin import algorithms

from state import\
    prototype_phrase_state,\
    prototype_encdec_state,\
    prototype_search_state,\
    prototype_state