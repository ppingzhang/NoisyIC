import hiddenlayer as h
import torch 

def save_graph(MyModel, save_name='model.png'):
	vis_graph = h.build_graph(MyModel, torch.zeros([1 ,3, 128, 128]))   
	vis_graph.theme = h.graph.THEMES["blue"].copy()     
	vis_graph.save(save_name)   

if __name__ == '__main__':
	import sys
	sys.path.append('../')
	from model.AutoEncoder import AutoEncoder
	from tensorboardX import SummaryWriter

	mymodel = AutoEncoder()
	input_ = torch.zeros([1 ,3, 128, 128])
	out = mymodel(input_)
	

	writer = SummaryWriter('log')

	mm = torch.mean(out)
	writer.add_scalar("Loss/train", mm, 0)

	with writer as w:
	    w.add_graph(mymodel, (torch.zeros([1 ,3, 128, 128]),))

	writer.close()
	    