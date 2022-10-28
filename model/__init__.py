import importlib
import logging
logger = logging.getLogger('base')


def create_model(opt):

	#try:
	pkg = importlib.import_module('model.'+opt.model)
	M =  getattr(pkg, opt.model)

	#except:
	#	raise NotImplementedError('Model [{:s}] not recognized.'.format(opt.model))
		
	m = M(opt)
	logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
	return m
