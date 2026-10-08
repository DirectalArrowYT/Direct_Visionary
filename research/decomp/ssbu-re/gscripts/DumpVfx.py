# Decompile every function listed in vfx_targets.txt (see ../vfx_targets.py) into vfx_decomp.txt.
# Script arguments, both optional: the targets file, then the output file.
# Jython rather than Java: Ghidra 10.1's script bundles do not resolve on a newer JDK.
from ghidra.app.decompiler import DecompInterface
from java.lang import Exception as JavaException

starts, strings = [], []
args = getScriptArgs()
targets = args[0] if len(args) > 0 else "vfx_targets.txt"
output = args[1] if len(args) > 1 else "vfx_decomp.txt"
for line in open(targets):
    kind, addr = line.split()
    (starts if kind == "F" else strings).append(int(addr, 16))

for s in strings:
    try:
        createAsciiString(toAddr(s))
    except (Exception, JavaException):
        pass
for s in starts:
    a = toAddr(s)
    if getInstructionAt(a) is None:
        disassemble(a)
    if getFunctionAt(a) is None:
        try:
            createFunction(a, None)
        except (Exception, JavaException):
            pass

dec = DecompInterface()
dec.openProgram(currentProgram)
out = open(output, "w")
done = failed = 0
for s in starts:
    f = getFunctionAt(toAddr(s))
    if f is None:
        failed += 1
        continue
    res = dec.decompileFunction(f, 120, monitor)
    out.write("\n// ===== %s size %d =====\n" % (f.getEntryPoint(), f.getBody().getNumAddresses()))
    if res is not None and res.decompileCompleted():
        out.write(res.getDecompiledFunction().getC())
        done += 1
    else:
        out.write("// decompile failed\n")
        failed += 1
out.close()
print("DumpVfx done %d failed %d" % (done, failed))
