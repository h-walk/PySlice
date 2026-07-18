import os,sys,shutil
import matplotlib.pyplot as plt

# ensure we have pySEA downloaded
if not os.path.exists("../../pySEA"):
    print("downloading pySEA from private github repo")
    u,k=open("/home/qwe/.gitp").readlines() # may need git username and git key for private repo
    u=u.strip() ; k=k.strip()
    os.system("git clone https://"+u+":"+k+"@github.com/sea-ecosystem/sea-eco")
    shutil.move("sea-eco/src/pySEA","../../")
    shutil.rmtree("sea-eco")

# make sure the import works before we try anything else....
sys.path.insert(1,"../../")
from pySEA.sea_eco.io import load
from pySEA.sea_eco.architecture.base_structure_numpy import SEAFile

# run 05, which should conditionally save a .sea file if pySEA is found
if not os.path.exists("04_haadf.sea"):
    print("sea file does not exist, running 04_haadf")
    os.system("python3 04_haadf.py")

# run 05, which should conditionally save a .sea file if pySEA is found
if not os.path.exists("05_tacaw.sea"):
    print("sea file does not exist, running 05_tacaw")
    os.system("python3 05_tacaw.py")

# attempt to reload and plot it
loaded = load(file_path='05_tacaw.sea')
loaded.show_tree()

loaded.show(dims=('frequency','kx'))
plt.show()

loaded.show(dims=('kx','ky'))
plt.show()

loaded = load(file_path='04_haadf.sea')
loaded.show()
plt.show()
