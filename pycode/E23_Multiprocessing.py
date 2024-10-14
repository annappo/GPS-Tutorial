# Example 23: Multiprocessing (load and run E23_Multiprocessing.py e.g. in IDLE or use "python <name>" in Terminal)
# SPDX-FileCopyrightText: Copyright (C) 2023 Andreas Naber <annappo@web.de>
# SPDX-License-Identifier: GPL-3.0-only

import multiprocessing as mp
import time
import numpy as np
import gpslib_tutorial as gpslib

# default values for instances of satStream()
N_CYC = 32                 
CODE_SAMPLES = 2048        
NGPS = N_CYC*CODE_SAMPLES
SAMPLE_RATE = 2.048E6    
CORR_MIN = 8
CORR_AVG = 8
SMP_TIME = 0 

MAX_SAT = 8
noCalls = 3


# ------- Multiprocessing functions --------------

# this function is executed in parallel processes spawned by multiprocessing
def runProc(inQ,outQ):
    while True:
        msg = inQ.get()         # wait for instruction and data
        
        if msg[0]=='initPool':
            global WORKER_NO
    
            WORKER_NO = msg[1]
            process = mp.current_process()    
            name = process.name
            outQ.put((name,WORKER_NO))            
            
        elif msg[0]=='initInst':
            global SATPROC
            
            satNo,freq,delay,itSweep = msg[1]
            SATPROC = gpslib.SatStream(satNo,freq,delay=delay,itSweep=itSweep)
            outQ.put(satNo)            
                        
        elif msg[0]=='runInst':            
            data,smpTime = msg[1]
            swFq,frameData,coPh,cpQ = SATPROC.process(data,smpTime)
            satNo = SATPROC.SAT_NO
            outQ.put((swFq,satNo,frameData,coPh,cpQ,WORKER_NO))
        
        elif msg[0]=='done':
            break

        
def initMultiProcPool(poolNo):   
    pool = []
    for workerNo in range(poolNo):
        inQ = mp.Queue()
        outQ = mp.Queue()
        satProc=mp.Process(target=runProc, args=(inQ,outQ,),
                           name='worker%d' % (workerNo,))
        satProc.start()
        pool.append((inQ,outQ,satProc))
    
    for workerNo,(inQ,_,_) in enumerate(pool):
        inQ.put(('initPool',workerNo))

    resLst = []
    for _,outQ,_ in pool:
        name,no = outQ.get()
        resLst.append((name,no))

    return pool,resLst


def initPoolStreams(pool,allSats):

    initSats = []
    for poolNo in range(len(pool)):
        freq,delay,itSweep = poolNo*10,0,20                # arbitrary data
        inQ,outQ,_ = pool[poolNo]
        inQ.put(('initInst',(allSats[poolNo],freq,delay,itSweep))) 
        satNo = outQ.get()
        initSats.append(satNo)        
                                   
    return initSats


def closeMultiProcPool(pool):        
    for inQ,outQ,satProc in pool:
        inQ.put(('done',None))
        satProc.join()
        satProc.close()


# ---- main -------------------------------------
 
if __name__ == '__main__':    
    SMP_TIME = 0
    allSats = [i for i in range(1,33)]
    
    mp.set_start_method('spawn')   # default in POSIX is 'fork' 
    pool,resLst = initMultiProcPool(MAX_SAT)
    
    for name,no in resLst:
        print('Name: %s, Number: %d' % (name,no))
    print()
    
    try:        
        initSats = initPoolStreams(pool,allSats)
        print(initSats)
        print()

        for _ in range(noCalls):
            SMP_TIME += 32*NGPS  # 1 sec; frame list sent only once per second           
            data = np.asarray([np.random.rand()+1j*np.random.rand()\
                               for x in range(NGPS)],dtype = np.complex64)

            pnoLst=[]
            for poolNo in range(len(pool)):
                inQ,outQ,_ = pool[poolNo]
                inQ.put(('runInst',(data,SMP_TIME)))
                pnoLst.append(outQ)

            t1 = time.perf_counter()            
            resLst=[]
            for outQ in pnoLst:
                resLst.append(outQ.get())
            t2 = time.perf_counter()
                
            for swFq,satNo,fLst,coPh,cpQ,workNo in resLst:
                freq = fLst[0]['FRQ'] if len(fLst)>0 else 0.0
                print('sat %d, worker %d, freq %1.1f' % (satNo,workNo,freq))
            print('execution time: %1.4f sec' % (t2-t1))
            print()
                        
    finally:
        closeMultiProcPool(pool)
                
