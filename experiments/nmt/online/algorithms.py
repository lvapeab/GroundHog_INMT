import logging
from groundhog.trainer.SGD_adagrad import AdaGrad
from groundhog.trainer.SGD_online import SGD

logging.basicConfig(level=logging.DEBUG, format=" %(msecs)d: %(message)s")

logger = logging.getLogger(__name__)
def loadAlgorithm(state, model, data, algorithm_name, word_probs=None, seqlen=None, iterations=None,
                  tolerance=None, iterate_until_convergence=None, verbosity=0):

    logger.info('Computing algorithm: %s' % algorithm_name)

    if seqlen is None:
        seqlen = state['seqlen']

    if algorithm_name == 'PassiveAggressive':
        raise NotImplementedError, 'PassiveAggressive is still not implemented'
        # PA accepts extra parameters
        algorithm = eval(algorithm_name)(model, state, batch_iter,
                                    word_probs, seqlen, iterations,
                                    tolerance, iterate_until_convergence)
    else:
        logger.info('Model: %s'%str(model))
        logger.info('Algorithm: %s'%str(algorithm_name))
        algorithm = eval(algorithm_name)(model, state, data)

    return algorithm
