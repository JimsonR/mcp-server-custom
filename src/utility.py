import os
import sys

def add_site_packages_to_sys_path():
    site_packages = os.path.join(os.getcwd(),'.python_packages','lib', 'site-packages')
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)

