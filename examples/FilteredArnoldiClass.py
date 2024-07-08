from ngsolve import BaseMatrix, Matrix, MultiVector, Norm, InnerProduct, TaskManager
import scipy.linalg as spl
import numpy as np
import time as ti
from math import sqrt, pi, cos


class FilteredPi(BaseMatrix):
    def __init__(self,mata,matm_inv,tau,steps,filterf,freedofs=None):
        super().__init__()
        self.dt = tau
        self.filter = filterf
        self.nsteps = steps
        
        self.mata = mata
        self.matm_inv = matm_inv
        
        self.freedofs=freedofs
        
        
        self.vecE = self.mata.CreateColVector()
        self.tmpvec1 = self.mata.CreateColVector()
        self.tmpvec2 = self.mata.CreateColVector()

    # def Height(self):
    #    return self.mata.height
    #def Width(self):
    #    return self.mata.width
    
    # def CreateRowVector(self):
    #    return self.mata.CreateRowVector()
    def CreateColVector(self):
        return self.mata.CreateColVector()
    
    def Shape(self):
        return self.mata.shape
    def CreateVector(self,col):
        return self.mata.CreateVector(col)
    
    def Mult(self,rhs,out):
        with TaskManager():
            # self.SetInitialValues(rhs)
            self.vecE.data = rhs
            tau = self.dt
            out.data = tau*self.filter(0)*self.vecE
            t = 0
            
            unew = self.tmpvec1
            uold = self.tmpvec2
            uold.data = self.vecE
            
            with TaskManager():
                for i in range(self.nsteps):
                    t += tau       
                    print("\r time = {}, step = {}".format(t,i),end="")

                    unew.data = 2*self.vecE - uold 
                    unew.data -= tau**2 * self.matm_inv@self.mata * self.vecE
                    if self.freedofs:
                        unew.data[~self.freedofs] = 0.
                    uold.data = self.vecE
                    self.vecE.data = unew.data

                    out.data += tau*self.filter(t)*self.vecE

    def MultMulti(self,rhs,out):
        #experimental, not really faster yet
        self.vecEMulti = MultiVector(self.vecE.size,len(rhs),self.mata.is_complex)
        self.tmpvec1Multi = MultiVector(self.vecE.size,len(rhs),self.mata.is_complex)
        self.tmpvec2Multi = MultiVector(self.vecE.size,len(rhs),self.mata.is_complex)
        with TaskManager():
            self.vecEMulti = rhs.Evaluate()
            tau = self.dt
            #print(tau*self.filter(0)*self.vecEMulti)
            out = tau*self.filter(0)*self.vecEMulti
            t = 0
            
            unew = self.tmpvec1Multi
            uold = self.tmpvec2Multi
            uold = self.vecEMulti.Evaluate()
            for i in range(self.nsteps):
                t += tau       
                print("\r time = {}, step = {}".format(t,i),end="")

                unew = (2*self.vecEMulti - uold).Evaluate()
                unew -= (tau**2 * self.matm_inv@self.mata * self.vecEMulti.Evaluate()).Evaluate()
                uold = self.vecEMulti
                self.vecEMulti = unew
                
                out += (tau*self.filter(t)*self.vecEMulti).Evaluate()
    # def Reset(self):
    #    self.vecE[:]=0
    #
    # def SetInitialValues(self,vec):
    #    self.vecE.data = vec


def FilterFunction(lam, filterf, tau):
    nsteps = len(filterf)-1
    x = np.ones(lam.shape)
    z = np.ones(lam.shape)
    out = 1/2*tau*filterf[0]*np.ones(lam.shape)
    for i in range(nsteps):
        y = 2*x - z - tau**2 * lam * x
        z = x
        x = y
        out += tau*filterf[i+1]*x
    out += tau/2*filterf[i+1]*x
    return out

def FilterFunctionC(lam, filterf, tau):
    nsteps = len(filterf)-1
    x = np.ones(lam.shape)
    z = np.ones(lam.shape)
    out = tau*filterf[0]*np.ones(lam.shape)+0j
    for i in range(nsteps):
        y = 2*x - z - tau**2 * lam * x
        z = x
        x = y
        out += tau*filterf[i+1]*x
    return out

def FilterFunction_w(w, filterf, tau):
    nsteps = len(filterf)-1
    x = np.ones(w.shape)
    z = np.ones(w.shape)
    out = tau/2*filterf[0]*np.ones(w.shape)
    for i in range(nsteps):
        y = 2*x - z - tau**2 * w**2 * x
        z = x
        x = y
        out += tau*filterf[i+1]*x
    out -= tau/2*filterf[i+1]*x
    return out

class BlockKrylow:
    def __init__(self,mat,L,ipmat = None, startvector = None, freedofs = None):
        self.mat = mat
        self.abv = MultiVector(mat.Shape()[0],0,mat.is_complex)
        self.tmp = mat.CreateColVector()
        self.i = 0
        self.L = L
        self.ipmat = ipmat
        self.ApplyTime = 0.
        self.OrthoTime = 0.
        self.freedofs = freedofs
        if not startvector:
            tmpvec = mat.CreateColVector()
            for i in range(L):
                tmpvec.SetRandom()
                if self.freedofs:
                    tmpvec[~self.freedofs] = 0.
                
                now = ti.time()
                self.abv.AppendOrthogonalize(tmpvec,ipmat = self.ipmat)
                self.OrthoTime += ti.time()-now
            
        else:
            for i in range(L):
                now = ti.time()
                self.abv.AppendOrthogonalize(startvector[i], ipmat = self.ipmat)
                self.OrthoTime += ti.time()-now

    def Step(self,niter = 1): 
        self.i+=1
        for j in range(self.L):
            now = ti.time()
            self.tmp.data = self.mat*self.abv[-self.L]
            self.ApplyTime += ti.time()-now
            now = ti.time()
            self.abv.AppendOrthogonalize(self.tmp,ipmat = self.ipmat,iterations = niter)
            self.OrthoTime += ti.time()-now


class FilteredBlockKrylow(BlockKrylow):
    def __init__(self,mat,L,mats,matm,matm_inv,orthom = True, startvector = None,freedofs = None):
        self.mats = mats
        self.matm = matm
        self.matm_inv = matm_inv
        self.ProjectTime = 0.
        self.ResTime = 0.
        self.orthom = orthom
      
        if orthom:
            super().__init__(mat,L,self.matm, startvector,freedofs)
     
        else:
            super().__init__(mat,L,None, startvector,freedofs)

    def SolveProjectedInitialProblem(self):
        now = ti.time()
        tvecs = MultiVector(self.abv[0].size,len(self.abv),self.mats.is_complex)
        tvecs.data = self.mats*self.abv
        tvecs2 = MultiVector(self.abv[0].size,len(self.abv),self.mats.is_complex)
        #tvecs2.data = self.matm_inv*tvecs
        tvecs2.data = self.matm*self.abv
   
        #mats_proj = InnerProduct(tvecs,tvecs2)
        #matm_proj = InnerProduct(tvecs,self.abv)
        mats_proj = InnerProduct(tvecs,self.abv)
        #lam,v = spl.eig(mats_proj.NumPy(),matm_proj.NumPy())
        if self.orthom:
            lam,v = spl.eigh(mats_proj.NumPy())
        else:
            matm_proj = InnerProduct(tvecs2,self.abv)
            lam,v = spl.eigh(mats_proj.NumPy(),matm_proj.NumPy())
            
        #print(lam)
        #print("mats")
        #print(mats_proj)
        #print("matm")
        #print(matm_proj)
        self.ProjectTime += ti.time()-now
        return lam.real,(self.abv*Matrix(v.real)).Evaluate()

    def ComputeInitialResiduals(self,lam,v):
        now = ti.time()
        res = MultiVector(v[0].size,len(v),self.matm.is_complex)
        for i in range(len(v)):
            res[i].data = (self.mats-lam[i]*self.matm)*v[i]
            if self.freedofs:
                res[i][~self.freedofs] = 0.
        

        tvecs = MultiVector(v[0].size,len(v),self.matm.is_complex)
        tvecs.data = self.matm_inv*res
        #tvecs2 = MultiVector(v[0].size,len(v),self.matm.is_complex)
        #tvecs2.data = self.matm*v
        #resscal = np.array([sqrt(abs(InnerProduct(tvecs[i],res[i]))/abs(InnerProduct(tvecs2[i],v[i]))) for i in range(len(v))])
        resscal = np.array([sqrt(abs(InnerProduct(tvecs[i],res[i]))) for i in range(len(v))])
        reseuclid = np.array([sqrt(abs(InnerProduct(res[i],res[i]))) for i in range(len(v))])
        resinv = np.array([sqrt(abs(InnerProduct(tvecs[i],tvecs[i]))) for i in range(len(v))])

        self.ResTime += ti.time()-now
        return resscal,reseuclid,resinv

    def PrintTimes(self):
        print("ApplyTime = {} \n OrthogonalizeTime = {} \n ProjectTime = {} \n ResTime = {} \n total time = {}".format(
            self.ApplyTime,self.OrthoTime, self.ProjectTime, self.ResTime, 
            self.ApplyTime+self.OrthoTime+ self.ProjectTime+ self.ResTime  )) 


class BlockArnoldi:
    def __init__(self,mat, K, L, startvector = None):
        self.mat = mat
        self.H = Matrix(K*L,K*L, complex=mat.is_complex)
        self.H[:,:] = 0
        self.abv = MultiVector(mat.Shape()[0],0,mat.is_complex)
        self.L = L
        self.K = K
        self.i = 0
        if not startvector:
            self.v = MultiVector(mat.Shape()[0],L,mat.is_complex)
            for vi in self.v:
                vi.SetRandom()
        else:
            self.v = startvector
        self.v.Orthogonalize()

    def Step(self): 
        for vi in self.v:
            self.abv.Append(vi)
        self.v = (self.mat*self.v).Evaluate()
        #could be faster?
        #self.mat.MultMulti(self.v,self.v)
        L = self.L 
        for j in range(self.i+1):
            self.H[j*L:(j+1)*L,self.i*L:(self.i+1)*L] = InnerProduct(self.abv[L*j:(j+1)*L],self.v)
            self.v =(self.v- (self.abv[j*L:(j+1)*L]*self.H[j*L:(j+1)*L,self.i*L:(self.i+1)*L])).Evaluate()
        if self.i+1 < self.K:
            self.H[(self.i+1)*L:(self.i+2)*L,self.i*L:(self.i+1)*L] = self.v.Orthogonalize()
        self.i+=1

    def SolveHessenberg(self):
        L = self.L
        lam,ev = spl.eig(self.H[:L*(i+1),:L*(i+1)])
        ids = lam.argsort()[::-1]
        lam = lam[ids]
        ev = ev[:,ids]
        return lam, (self.abv*Matrix(ev.real)).Evaluate()

class FilteredBlockArnoldi(BlockArnoldi):
    def __init__(self,mat,K,L,mats,matm,matm_inv,startvector = None, filename = None):
        self.mats = mats
        self.matm = matm
        self.matm_inv = matm_inv
        self.filename = filename
        if filename:
            with open(filename,"w") as f:
                f.write("# K={}, L={}\n".format(K,L))

        super().__init__(mat,K,L,startvector)

    def SolveProjectedInitialProblem(self,niter = 1):
        print("solve projected initial problem orthogonalizing {} times".format(niter))
        for i in range(niter-1):
            self.abv.Orthogonalize()


        tvecs = MultiVector(self.abv[0].size,len(self.abv),self.mats.is_complex)
        tvecs.data = self.mats*self.abv
        tvecs2 = MultiVector(self.abv[0].size,len(self.abv),self.mats.is_complex)
        #tvecs2.data = self.matm_inv*tvecs
        tvecs2.data = self.matm*self.abv
        
        #mats_proj = InnerProduct(tvecs,tvecs2)
        #matm_proj = InnerProduct(tvecs,self.abv)
        mats_proj = InnerProduct(tvecs,self.abv)
        matm_proj = InnerProduct(tvecs2,self.abv)
        
        lam,v = spl.eigh(mats_proj.NumPy(),matm_proj.NumPy())
        return lam,(self.abv*Matrix(v.real)).Evaluate()

    def ComputeInitalEvs(self,v):
        tvecs = MultiVector(self.v[0].size,len(v),self.mats.is_complex)
        tvecs = (self.mats*v).Evaluate()
        tvecs = (self.matm_inv*tvecs).Evaluate()
        lams = []
        for i in range(len(v)):
            lams.append(InnerProduct(v[i],tvecs[i]))
            lams[-1]/=InnerProduct(v[i],v[i])
        return lams
        

    def ComputeInitialResiduals(self,lam,v,normalize=True):
        res = MultiVector(v[0].size,len(v),self.matm.is_complex)
        for i in range(len(v)):
            res[i].data = (self.mats-lam[i]*self.matm)*v[i]
        

        tvecs = MultiVector(v[0].size,len(v),self.matm.is_complex)
        tvecs = (self.matm_inv*res).Evaluate()

        if normalize:
            tvecs2 = MultiVector(v[0].size,len(v),self.matm.is_complex)
            tvecs2 = (self.matm*v).Evaluate()
            resscal = np.array([sqrt(abs(InnerProduct(tvecs[i],res[i]))/abs(InnerProduct(tvecs2[i],v[i]))) for i in range(len(v))])
        else:
            resscal = np.array([sqrt(abs(InnerProduct(tvecs[i],res[i]))) for i in range(len(v))])

        return resscal,res

    def ComputeFilteredResiduals(self,lam,v):
        pass






def TD_EVP(mata,matm,matm_inv,tau,steps,filterf=None,blocksize=1,maxsteps=50,startvec=None,filename=None):
        if not filterf:
                omega0 = sqrt(2)
                sigma = 1
                filterf = lambda t: 4/pi*cos(omega0*t)*sigma*np.sinc(sigma*t/pi)
 
        Pi = FilteredPi(mata,matm_inv,tau,steps,filterf)
        print("ndof: {}".format(Pi.mata.height))

        now = ti.time()

        fba = FilteredBlockArnoldi(Pi,maxsteps,blocksize,mata,matm,matm_inv,startvec, filename)
        for i in range(maxsteps):
            print("Arnoldistep = {}".format(i))
            fba.Step()
        lam,vecs = fba.SolveProjectedInitialProblem()
        

        arnolditime = ti.time()-now
        print("Evs: {}".format(lam))
        return lam,vecs,arnolditime










