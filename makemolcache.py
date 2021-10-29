import pickle
import molgrid
import numpy as np
import os,sys
import struct, traceback
import argparse
import multiprocessing

mols_to_read = multiprocessing.Queue()
mols_to_write = multiprocessing.Queue()
N = multiprocessing.cpu_count() * 2

map_to_smina_types={0.0:0, 1.0:2,2.0:6,3.0:10}
def read_data():
      while True:
        sys.stdout.flush()
        (idx, coords,types) = mols_to_read.get()
        if idx == None:
            break
        try:
              mapped_types = np.vectorize(map_to_smina_types.get)(types)
              values = np.concatenate((coords,mapped_types[np.newaxis,:].T),axis=1)#.flatten().tolist()
              data = b''
              for row in values:
                  data += struct.pack('fffi',row[0],row[1],row[2],int(row[3]))
              #data = struct.pack(f'{len(values)}f',*values)
              assert(len(data) % 16 == 0)
              if len(data) == 0:
                  print(idx,"EMPTY")
              else:
                  mols_to_write.put((idx,data))
        except Exception as e:
              print(idx,mol)
              print(e)
      mols_to_write.put(None)

def fill_queue(mols):
    for idx, (coords, types, _, _) in enumerate(mols):
        mols_to_read.put((idx,coords,types))
    for _ in range(N):
        mols_to_read.put((None,None,None))


def create_cache2(molfiles, outfile):
    out = open(outfile,'wb')

    out.write(struct.pack('i',-1))
    out.write(struct.pack('L',0))

    filler = multiprocessing.Process(target=fill_queue, args=(molfiles,))
    filler.start()

    readers = multiprocessing.Pool(N)
    for _ in range(N):
        readers.apply_async(read_data)

    offsets = dict()
    endcnt = 0
    while True:
        moldata = mols_to_write.get()
        if moldata == None:
            endcnt += 1
            if endcnt == N:
                break
            else:
                continue

        (idx, data) = moldata
        offsets[f'mol-{idx}'] = out.tell()
        natoms = len(data) // 16
        out.write(struct.pack('i',natoms))
        out.write(data)

    start = out.tell()
    for idx, _ in enumerate(molfiles):
        if f'mol-{idx}' not in offsets:
            print(f"skipping {idx} since it failed to read in")
            continue
        s = bytes(f'mol-{idx}', encoding='UTF-8')
        out.write(struct.pack('B',len(s)))
        out.write(s)
        out.write(struct.pack('L',offsets[f'mol-{idx}']))

    out.seek(4)
    out.write(struct.pack('L',start))
    out.seek(0,os.SEEK_END)
    out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle','-P',required=True,help='location of pickle file with data')
    parser.add_argument('--traincache','-T',default='pickle',help='name of output training molcache file')
    parser.add_argument('--testcache','-E',default='pickle',help='name of output testing molcache file')
    args = parser.parse_args()


    (train, test) = pickle.load(open(args.pickle,'rb'))

    create_cache2(train+test,args.traincache)
