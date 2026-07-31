# NetCDF Mesh Importer for Blender 5

A Blender add-on that allows you to import mesh data from NetCDF (.nc) files into Blender 5.

## Features

- **Import**: Load NetCDF files containing mesh/grid data into Blender
- **Export**: Save edited Blender meshes back to NetCDF format
- Supports UGRID conventions for unstructured grids
- Automatically detects vertices, edges, and faces
- Handles 2D and 3D coordinate data
- Creates Blender mesh objects from NetCDF data
- Validates mesh data to prevent crashes

## Requirements

- Blender 5.0 or later
- xarray Python module (already installed in your Blender's bundled Python)
- numpy (typically included with xarray)

## Installation

1. Copy the `netcdf_importer` folder to your Blender add-ons directory:
   - **Windows**: `%APPDATA%\Blender Foundation\Blender\5.0\scripts\addons\`
   - **macOS**: `~/Library/Application Support/Blender/5.0/scripts/addons/`
   - **Linux**: `~/.config/blender/5.0/scripts/addons/`

2. Alternatively, you can install it directly:
   - Open Blender
   - Go to `Edit > Preferences > Add-ons`
   - Click `Install...` and select the `netcdf_importer` folder
   - Enable the add-on by checking the box next to "NetCDF Mesh Importer"

## Usage

### Importing

1. In Blender, go to `File > Import > NetCDF Mesh (.nc)`
2. Select your NetCDF file (e.g., `Grid_002_net.nc`)
3. Adjust import options if needed:
   - **Use UGRID Convention**: Enable for standard UGRID files
   - **Auto Scale Coordinates**: Automatically scale small coordinates
   - **Coordinate Scale Factor**: Manual scaling factor
4. The mesh will be imported and added to your scene

### Exporting

1. Select the mesh object you want to export
2. Go to `File > Export > NetCDF Mesh (.nc)`
3. Choose a filename and location
4. Adjust export options:
   - **Use UGRID Convention**: Export following UGRID standards (recommended)
   - **Use 1-Based Indexing**: Use 1-based indexing for connectivity (standard for UGRID)
   - **Include Edges**: Export edge connectivity information
5. Click "Export NetCDF Mesh" to save the file

## Supported Formats

The add-on supports:

- **UGRID Convention**: Standard NetCDF format for unstructured grids
  - Looks for `node_x`, `node_y`, `node_z` or `lon`, `lat`, `depth` coordinates
  - Detects `face_node_connectivity` and `edge_node_connectivity` arrays

- **Generic Formats**: 
  - Variables named `vertices`, `nodes`, `points`, or `coordinates`
  - Variables named `edges` or `connectivity`
  - Variables named `faces` or `triangles`

## File Structure

```
netcdf_importer/
├── __init__.py          # Add-on registration
└── netcdf_importer.py   # Main import operator
```

## Troubleshooting

- **xarray not found**: Make sure xarray is installed in Blender's bundled Python. You can verify this by opening Blender's Python console and running `import xarray`.
- **No mesh created**: Check that your NetCDF file contains recognizable coordinate and connectivity variables. The add-on will report errors in Blender's info area.
- **Incorrect mesh**: Try toggling the "Use UGRID Convention" option in the import dialog.

## Notes

### Import
- The add-on automatically detects 0-based vs 1-based indexing
- Missing Z coordinates will be set to 0.0
- Face connectivity arrays with fill values (typically -1) are automatically handled
- Mesh validation prevents crashes from invalid geometry

### Export
- Exported files follow UGRID conventions and can be read by Delft3D/RFGRID
- By default, uses 1-based indexing (standard for UGRID)
- Face connectivity arrays are padded with fill values (-1) for faces with fewer vertices
- Edge connectivity is optional and can be excluded if not needed
- Files are compressed for smaller file sizes
