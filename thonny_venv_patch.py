import sys
import site

# Path to your venv's site-packages
venv_site = "/home/morgan/dogbot/.venv/lib/python3.11/site-packages"

if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
    site.addsitedir(venv_site)
