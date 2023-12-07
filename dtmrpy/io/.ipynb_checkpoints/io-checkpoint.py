import gzip
import tempfile
import shutil
import os
from subprocess import Popen
import nibabel as nib
from scipy.io import loadmat

def gz_to_mat(file_to_convert,save_folder=""):
    with gzip.open(file_to_convert, 'rb') as file_in:
        with open(save_folder+file_to_convert.split('\\')[-1]+'.mat', 'wb') as file_out:
            shutil.copyfileobj(file_in, file_out)   
            
def loadsrc(src_filename,data_folder=''):
    with tempfile.TemporaryDirectory() as temp_folder:
        gz_to_mat(src_filename,temp_folder)    
        return loadmat(temp_folder+src_filename.split('\\')[-1])
    
def loadfib(fib_filename,data_folder=''):
    with tempfile.TemporaryDirectory() as temp_folder:
        gz_to_mat(fib_filename,temp_folder)    
        return loadmat(temp_folder+fib_filename.split('\\')[-1])
    
def loadtdi(fib_filename, tt_filename):
    with tempfile.TemporaryDirectory() as temp_folder:
        #get tdi files - run analysis on fib file
        dsi_path = r"path=C:\Users\micha\Desktop\Other\Random Shit\New folder\Diffusion MRI_old_maybe\dsi_studio_win"+'\n'
        command = 'dsi_studio --action=ana --source="'+fib_filename+'" --tract="'+tt_filename+'" --output="'+temp_folder+'tract" --export=tdi'+' > log.txt"'

        bat_temp = temp_folder+"tract.bat"

        #Write .bat file
        with open(bat_temp, "w") as f:
            f.writelines(dsi_path)
            f.writelines(command)

        p = Popen(bat_temp,cwd=temp_folder+'\\')
        stdout, stderr = p.communicate()
        os.remove(bat_temp)

        return nib.load(temp_folder+'tract.tdi.nii.gz').get_fdata()
    
    
    
def generate_tt(fib_filename, tract, tt_filename):
    with tempfile.TemporaryDirectory() as temp_folder:
        #get tdi files - run analysis on fib file
        dsi_path = r"path=C:\Users\micha\Desktop\Other\Random Shit\New folder\Diffusion MRI_old_maybe\dsi_studio_win"+'\n'
        # command = 'dsi_studio --action=atk --source="'+fib_filename+'" --track_id="'+tract.strip()+'" --output="'+tt_filename+'"'+' > log.txt"'
        command = 'dsi_studio --action=trk --source="'+fib_filename+'" --track_id="'+tract.strip()+'" --output="'+tt_filename+'"'

        bat_temp = temp_folder+"tract.bat"

        #Write .bat file
        with open(bat_temp, "w") as f:
            f.writelines(dsi_path)
            f.writelines(command)

        p = Popen(bat_temp,cwd=temp_folder+'\\')
        stdout, stderr = p.communicate()
        os.remove(bat_temp)

        # return 'C:\\Users\\micha\\Desktop\\New folder (3)'+'\\'+tt_filename