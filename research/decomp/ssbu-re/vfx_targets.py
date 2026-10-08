"""List the function starts and referenced strings in a .text range, for `gscripts/DumpVfx.py`.

A raw import has no analysis to mark either. Function starts are every `bl` in .text that lands
in the range plus every relocated data pointer into it (vtable slots); strings are the rodata
addresses the range's own adrp+add pairs form. Defaults to where NintendoWare Vfx sits in 13.0.3.

usage: vfx_targets.py [start end]   ->  vfx_targets.txt  ("F addr" / "S addr" lines)
"""
import struct, sys
import numpy as np
from dump_paths import dump_file
img=dump_file('main_reloc.bin').read_bytes()
start=int(sys.argv[1],0) if len(sys.argv)>1 else 0x60000
end=int(sys.argv[2],0) if len(sys.argv)>2 else 0xd0000
ro=struct.unpack_from('<I',dump_file('exefs/main').read_bytes(),0x24)[0]  # .rodata memory offset = end of .text
w=np.frombuffer(img,dtype='<u4',count=ro//4).astype(np.int64)
pc=np.arange(len(w),dtype=np.int64)*4
bl=(w&0xfc000000)==0x94000000
imm=w[bl]&0x3ffffff; imm=np.where(imm&0x2000000,imm-0x4000000,imm)
tg=pc[bl]+imm*4
starts=set(int(t) for t in tg[(tg>=start)&(tg<end)])
q=np.frombuffer(img,dtype='<u8',offset=ro,count=(len(img)-ro)//8)
starts|=set(int(v) for v in q[(q>=start)&(q<end)&(q&3==0)])
strings=set(); page=[None]*32
for p in range(start,end,4):
    x=int(w[p//4])
    if x&0x9f000000==0x90000000:
        i=(((x>>5)&0x7ffff)<<2)|((x>>29)&3)
        if i&(1<<20): i-=1<<21
        page[x&31]=(p&~0xfff)+(i<<12)
    elif x&0xff800000==0x91000000 and page[(x>>5)&31] is not None:
        t=page[(x>>5)&31]+(((x>>10)&0xfff)<<(12*((x>>22)&1)))
        if ro<=t<len(img):
            z=img.find(b'\0',t,t+512)
            if z-t>=3 and all(32<=c<127 or c in(9,10) for c in img[t:z]): strings.add(t)
with open('vfx_targets.txt','w') as f:
    for s in sorted(starts): f.write('F %#x\n'%s)
    for s in sorted(strings): f.write('S %#x\n'%s)
print('functions',len(starts),'strings',len(strings))
