"""
NetCDF Importer for Blender 5
A Blender add-on to import NetCDF (.nc) files containing mesh data
"""

bl_info = {
    "name": "NetCDF Mesh Importer",
    "author": "Evan Kennedy",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "location": "File > Import > NetCDF Mesh (.nc)",
    "description": "Import mesh data from NetCDF files using UGRID conventions",
    "category": "Import-Export",
}

import bpy
from . import netcdf_importer

def register():
    netcdf_importer.register()

def unregister():
    netcdf_importer.unregister()

if __name__ == "__main__":
    register()
