import torch as xp
import time
a=xp.rand(100000,device='cuda')
b=xp.rand(1000,device='cuda')
start=time.time()
#c=xp.histogram(a,bins=b)
c=xp.zeros(len(b)-1,device='cuda')
mask = xp.zeros(len(a),device='cuda')
for i,(b1,b2) in enumerate(zip(b[:-1],b[1:])):
	mask *= 0
	mask[a>=b1]=1 ; mask[a>=b2]=0
	c[i] = xp.sum(mask)
print(time.time()-start)
