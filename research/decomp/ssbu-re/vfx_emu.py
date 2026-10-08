"""Run particle-library functions from `main_reloc.bin` under Unicorn.

The library's matrix code is NEON shuffles that are slow to read and easy to misread; calling
it with chosen inputs and comparing the result against candidate formulas settles a rotation
order or a handedness directly. `Machine` maps the image, a stack and a scratch heap, and
fills in the sine/cosine tables the code reaches through the SDK (they live in another module
and are null in a raw image).
"""
import math, struct
from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_LITTLE_ENDIAN, UcError
from unicorn.arm64_const import *
from dump_paths import dump_file

STACK, HEAP, RET = 0x7000000000, 0x8000000000, 0x9000000000
# GOT slot -> the float(s) the SDK would have put behind it.
MATH = {
    0x52a3740: [1/39916800, 1/362880, 1/5040, 1/120, 1/6],      # sine series, high order first
    0x52a3748: [1/3628800, 1/40320, 1/720, 1/24, 1/2],          # cosine series
    0x52a3750: [2*math.pi], 0x52a3758: [1/(2*math.pi)], 0x52a3760: [math.pi/2],
    0x52a3338: [math.pi], 0x52a37d8: [math.pi],
    # nn::util::Vector3f constants: Zero, UnitY, UnitX, UnitZ (in that slot order).
    0x52a7a88: [0, 0, 0, 0], 0x52a7a98: [0, 1, 0, 0], 0x52a7aa0: [1, 0, 0, 0], 0x52a7aa8: [0, 0, 1, 0],
}

class Machine:
    def __init__(self):
        img = dump_file('main_reloc.bin').read_bytes()
        self.mu = mu = Uc(UC_ARCH_ARM64, UC_MODE_LITTLE_ENDIAN)
        mu.mem_map(0, (len(img) + 0xfff) & ~0xfff); mu.mem_write(0, img)
        mu.mem_map(STACK, 0x100000); mu.mem_map(HEAP, 0x400000); mu.mem_map(RET, 0x1000)
        mu.reg_write(UC_ARM64_REG_CPACR_EL1, 3 << 20)            # enable FP/NEON
        self.top = HEAP + 0x300000
        for slot, values in MATH.items():
            at = self.alloc(len(values) * 4)
            mu.mem_write(at, struct.pack('<%df' % len(values), *values))
            mu.mem_write(slot, struct.pack('<Q', at))
    def alloc(self, size):
        at = self.top; self.top += (size + 0xf) & ~0xf; return at
    def write(self, at, fmt, *values): self.mu.mem_write(at, struct.pack('<' + fmt, *values))
    def read(self, at, fmt): return struct.unpack('<' + fmt, self.mu.mem_read(at, struct.calcsize('<' + fmt)))
    def call(self, addr, *args, floats=()):
        mu = self.mu
        for i, a in enumerate(args): mu.reg_write(UC_ARM64_REG_X0 + i, a)
        for i, f in enumerate(floats):
            mu.reg_write(UC_ARM64_REG_V0 + i, struct.unpack('<I', struct.pack('<f', f))[0])
        mu.reg_write(UC_ARM64_REG_SP, STACK + 0xf0000); mu.reg_write(UC_ARM64_REG_LR, RET)
        try: mu.emu_start(addr, RET, count=2_000_000)
        except UcError as e: raise RuntimeError('%s at pc=%#x' % (e, mu.reg_read(UC_ARM64_REG_PC)))
        return mu.reg_read(UC_ARM64_REG_X0)
