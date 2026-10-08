"""Apply the image's own relative relocations at base 0 and write `main_reloc.bin`.

`main_decompressed.bin` leaves every vtable and pointer table zeroed, because the loader fills
them in. With them applied, a raw import can follow virtual calls and data pointers.
"""
import struct
from dump_paths import dump_file
img=bytearray(dump_file('main_decompressed.bin').read_bytes())
def u32(o): return struct.unpack_from('<I',img,o)[0]
def u64(o): return struct.unpack_from('<Q',img,o)[0]
mod=u32(4)
assert img[mod:mod+4]==b'MOD0', "no MOD0 header"
dyn=mod+struct.unpack_from('<i',img,mod+4)[0]
tags={}
o=dyn
while True:
    tag,val=u64(o),u64(o+8); o+=16
    if tag==0: break
    tags[tag]=val
DT_RELA,DT_RELASZ,DT_JMPREL,DT_PLTRELSZ=7,8,23,2
R_RELATIVE=1027
applied=other=0
for off,size in ((tags.get(DT_RELA),tags.get(DT_RELASZ,0)),(tags.get(DT_JMPREL),tags.get(DT_PLTRELSZ,0))):
    if off is None: continue
    for e in range(off,off+size,24):
        r_off,r_info,r_add=struct.unpack_from('<QQq',img,e)
        if r_info&0xffffffff==R_RELATIVE:
            struct.pack_into('<Q',img,r_off,r_add); applied+=1
        else: other+=1
dump_file('main_reloc.bin').write_bytes(img)
print("relative relocations applied",applied,"symbolic left alone",other)
