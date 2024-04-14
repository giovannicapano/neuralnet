import numpy as np
import random as rand
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)



class Ising1d:
    def __init__(self,nspins_,hfield_,pbc_,mel_):
        self.nspins_=nspins_
        self.hfield_=hfield_
        self.pbc_=True
        self.mel=mel_
        
    
    def Init(self):
        self.mel_=torch.full((self.nspins_+1,),-self.hfield_, dtype=torch.float)
         
        print(self.mel_)
        

    def find_conn(self,state,mel):
        mel=self.mel_
        mel[0] = -torch.sum(state * torch.roll(state,1))
        if not self.pbc_:
           mel[0]+= 2*state[self.nspins_-1]*state[0]
        return mel
        

class wf(torch.nn.Module):
   def __init__(self,n_vis,n_hid,a,b,W):
      
      
      super(wf,self).__init__()
      self.n_vis=n_vis
      self.n_hid=n_hid
      self.a=a
      self.b=b
      self.W=W
      
   def logVal(self, state):
       thetah=0
       rbm=torch.dot(self.a,state)
       for h in range(self.n_hid):
        thetah=self.b[h]
        for v in range(self.n_vis):
          thetah=thetah+state[v]*self.W[v][h]
       rbm+=torch.log(torch.cosh(thetah))
       return rbm
   def pop_prob(self,state,flips):
      logpop=0
      for f in flips:
       logpop -= 2*self.a[f.int()]*state[f.int()]
      thetah = self.b + torch.matmul(state,self.W)
      """for v in range(self.n_vis_):
            thetah=thetah+state[v]*self.W_[v]"""
      thetahp=thetah
      for f in flips:
       thetahp -= 2*state[f.int()]*self.W[f.int()]
      logpop+=torch.sum(torch.log(torch.cosh(thetahp))-torch.log(torch.cosh(thetah)))
      return torch.exp(logpop)    
   def forward(self,state):
    logpop=0
    logpop = -2*torch.mul(self.a,state)
    
    thetah = (self.b + torch.matmul(state,self.W)).repeat(40,1)
    """for v in range(self.n_vis_):
          thetah=thetah+state[v]*self.W_[v]"""
    thetahp=thetah
    thetahp -= 2*torch.matmul(state,self.W)
    logpop+=torch.sum(torch.log(torch.cosh(thetahp))-torch.log(torch.cosh(thetah)))
    lg=logpop.tolist()
    lg.insert(0,0)
    logpop=torch.tensor(lg)
    return torch.exp(logpop)
    """return state[flips.int()]*self.a_[flips.int()]"""

class nqs(torch.nn.Module):
   def __init__(self,n_vis,n_hid,hamilton_):
    self.n_vis=n_vis
    self.n_hid=n_hid
    super(nqs,self).__init__()
    self.a=  nn.Parameter(torch.rand(n_vis, dtype=torch.float))
    self.b=  nn.Parameter(torch.rand(n_hid, dtype=torch.float))
    self.W=  nn.Parameter(torch.rand((n_vis,n_hid), dtype=torch.float))
    self.hamilton_=hamilton_ 
    self.wf=wf(self.n_vis,self.n_hid,self.a,self.b,self.W)
   def Prob(self,state,flips):
      return torch.linalg.norm(self.wf.pop_prob(state,flips))**2
   def forward(self,state,mel):
    en=0
    """for i in range(list(flipsh.size())[0]):"""
    en+=torch.matmul(self.wf(state),mel)
    return en
enn=[]
lr=1e-7
ising=Ising1d(40,0.5,True,0)   
ising.Init()
model=nqs(40,200,ising)
for n,p in model.named_parameters():
      print (n,p.shape)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr)
nspins_=40
state_=2*torch.randint(2, (nspins_,), dtype=torch.float)-1
s=state_.size()
print(s)
while torch.sum(state_)!=0:
 if(torch.sum(state_)<0):
    m=np.random.randint(0,nspins_)
    while state_[m]>0:  
      m=np.random.randint(0,nspins_)
    state_[m]=1
 if torch.sum(state_)>0 :
    m=np.random.randint(0,nspins_)
    while state_[m]<0:  
      m=np.random.randint(0,nspins_)
    state_[m]=-1
 
print(torch.sum(state_))

for t in range(100):
       flipp=torch.randint(0,nspins_,(1,))
       s=flipp[0].int()
       if(model.Prob(state_,flipp)>rand.uniform(0,1)):
        state_[s]=state_[s]*-1

def loop(mel,state_,model):
   en=Variable(torch.zeros(1), requires_grad=True)
   for t in range(500):
         flip=torch.randint(0,nspins_,(1,),dtype=torch.float)
         trial=torch.clone(state_)
         s=flip[0].int()
         trial[s]=trial[s]*-1
         """if(torch.sum(trial)<0):
         m=np.random.randint(0,nspins_)
         while trial[m]>0:  
            m=np.random.randint(0,nspins_)
         trial[m]=1
         if torch.sum(trial)>0 :
         m=np.random.randint(0,nspins_)
         while trial[m]<0:  
            m=np.random.randint(0,nspins_)
         trial[m]=-1"""
         """print(torch.sum(trial))"""
         if(model.Prob(state_,flip)>rand.uniform(0,1)):
            state_=torch.clone(trial)  
         mel=ising.find_conn(state_,mel)
         """print(mel)"""
         en_fin=model(state_,mel)
         en = torch.cat((en, en_fin.unsqueeze(0)))
         """optimizer.zero_grad()
         (en_fin).backward()
         optimizer.step()"""
   return en,state_   
     
enn=[]
mel=torch.full((nspins_,),0,dtype=torch.float)
for s in range(1000):
   for t in range(100):
            flipp=torch.randint(0,nspins_,(1,))
            s=flipp[0].int()
            if(model.Prob(state_,flipp)>rand.uniform(0,1)):
               state_[s]=state_[s]*-1
   en,state_=loop(mel,state_,model)
   print(state_)
   en_tot=torch.mean(en)
   enn.append(en_tot)
   print(en_tot)
   optimizer.zero_grad()
   en_tot.backward()
   """for g in optimizer.param_groups:
    g['lr'] = lr/(np.sqrt(s+1))
    print(g['lr'])"""
   optimizer.step()

"""print(enn)
print(sum(enn)/40000)
n_blocks=50
blocksize=np.intc(len(enn)/n_blocks)
enmean_unblocked=0
enmeansq_unblocked=0
enmean=0
enmeansq=0
for i in range(n_blocks):
    eblock=0
    j=(i*blocksize)
    if j<(i+1)*blocksize:

        eblock+=enn[j]

        delta=enn[j]-enmean_unblocked
        enmean_unblocked+=delta/(j+1)
        delta2=enn[j]-enmean_unblocked
        enmeansq_unblocked+=delta*delta2
        j+=1
    eblock=blocksize
    delta=eblock-enmean
    enmean+=delta/(i+1)
    delta2=eblock-enmean
    enmeansq+=delta*delta2
print(enmean/nspins_) """