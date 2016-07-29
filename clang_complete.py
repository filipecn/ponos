import os;

dirset = set()
for root, dirs, files in os.walk("."):
    for name in files:
        if name.endswith((".h", ".hpp")):
            dirset.add(root.strip('.'))

f = open('.clang_complete','w')
for dirname in dirset:
    f.write("-I" + dirname + "\n")
f.close()
