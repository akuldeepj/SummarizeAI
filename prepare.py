import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'flask'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'numpy'])
