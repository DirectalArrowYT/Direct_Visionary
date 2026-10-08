"""Disassemble a function from `main_reloc.bin`, labelling emitter-resource field offsets.

Ghidra's decompiler turns the library's NEON float code into byte-shuffling noise, so the
maths is easier to read as instructions. Loads and stores whose displacement matches an entry
in `emitter_offsets_v22.txt` get that field's name as a comment (a guess whenever the base
register is not the resource pointer), adrp+add/ldr targets are resolved, and float constants
and strings they point at are shown.

usage: vfx_asm.py <start> [end-or-size]
"""
import re, struct, sys
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
from dump_paths import dump_file
img=dump_file('main_reloc.bin').read_bytes()
names={}
try:
    for line in open('emitter_offsets_v22.txt'):
        if line.startswith('0x'):
            o,n=line.split(None,1); names[int(o,16)]=n.strip().split()[0]
except IOError: pass
start=int(sys.argv[1],0)
end=int(sys.argv[2],0) if len(sys.argv)>2 else 0
if end<start: end=start+(end or 0x400)
def const(a):
    if not 0<=a<len(img)-8: return ''
    z=img.find(b'\0',a,a+80)
    if z-a>=4 and all(32<=c<127 for c in img[a:z]): return '"%s"'%img[a:z].decode()
    f=struct.unpack_from('<f',img,a)[0]; u=struct.unpack_from('<I',img,a)[0]
    q=struct.unpack_from('<Q',img,a)[0]
    return 'f=%g u32=%#x q=%#x'%(f,u,q)
md=Cs(CS_ARCH_ARM64,CS_MODE_ARM); page={}
for ins in md.disasm(img[start:end],start):
    note=''
    op=ins.op_str
    if ins.mnemonic=='adrp':
        r,v=op.split(', '); page[r]=int(v.lstrip('#'),16)
    else:
        m=re.search(r'\[(x\d+), #(0x[0-9a-f]+|\d+)\]',op)
        m2=re.match(r'(x\d+), (x\d+), #(0x[0-9a-f]+|\d+)$',op) if ins.mnemonic=='add' else None
        if m2 and m2.group(2) in page:
            t=page[m2.group(2)]+int(m2.group(3),0); note='; %#x %s'%(t,const(t)); page[m2.group(1)]=t
        elif m and m.group(1) in page and ins.mnemonic.startswith('ldr'):
            t=page[m.group(1)]+int(m.group(2),0); note='; %#x %s'%(t,const(t))
        elif m and int(m.group(2),0) in names and int(m.group(2),0)>=0x40:
            note='; ?'+names[int(m.group(2),0)]
        elif m2 and int(m2.group(3),0) in names and int(m2.group(3),0)>=0x100:
            note='; ?&'+names[int(m2.group(3),0)]
        d=op.split(',')[0]
        if d in page and not m2 and ins.mnemonic in('mov','ldr','ldp','add','sub','movz','orr') and not (m and d==m.group(1)): page.pop(d,None)
    print('%07x  %-8s %-40s %s'%(ins.address,ins.mnemonic,op,note))
