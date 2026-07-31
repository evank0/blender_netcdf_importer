"""
NetCDF Importer Operator
Handles importing NetCDF files and creating Blender meshes
"""

import bpy
import os
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import StringProperty, BoolProperty, FloatProperty
from bpy.types import Operator

try:
    import xarray as xr
    import numpy as np
    from collections import defaultdict
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    print("Warning: xarray not available. NetCDF import may not work correctly.")


RGFGRID_FILL = -999
RGFGRID_TOPOLOGY_VALUE = np.int32(-2147483647)


def _rgfgrid_compute_edge_midpoints(node_x, node_y, edge_nodes, index_offset=1):
    idx0 = edge_nodes[:, 0] - index_offset
    idx1 = edge_nodes[:, 1] - index_offset
    edge_x = (node_x[idx0] + node_x[idx1]) / 2.0
    edge_y = (node_y[idx0] + node_y[idx1]) / 2.0
    return edge_x, edge_y


def _rgfgrid_compute_face_centroids(node_x, node_y, face_nodes, face_fill=RGFGRID_FILL, index_offset=1):
    n_faces = face_nodes.shape[0]
    face_x = np.zeros(n_faces, dtype=np.float64)
    face_y = np.zeros(n_faces, dtype=np.float64)
    for i in range(n_faces):
        valid = face_nodes[i] != face_fill
        if not np.any(valid):
            continue
        idx = face_nodes[i, valid] - index_offset
        xb = node_x[idx]
        yb = node_y[idx]
        if len(xb) >= 3:
            x0, y0 = xb[-1], yb[-1]
            area = cx_sum = cy_sum = 0.0
            for j in range(len(xb)):
                x1, y1 = xb[j], yb[j]
                cross = x0 * y1 - x1 * y0
                area += cross
                cx_sum += (x0 + x1) * cross
                cy_sum += (y0 + y1) * cross
                x0, y0 = x1, y1
            area *= 0.5
            if abs(area) > 1e-12:
                face_x[i] = cx_sum / (6.0 * area)
                face_y[i] = cy_sum / (6.0 * area)
            else:
                face_x[i] = np.mean(xb)
                face_y[i] = np.mean(yb)
        else:
            face_x[i] = np.mean(xb)
            face_y[i] = np.mean(yb)
    return face_x, face_y


def _rgfgrid_compute_face_bounds(node_x, node_y, face_nodes, face_fill=RGFGRID_FILL, index_offset=1):
    n_faces, n_max = face_nodes.shape
    bounds_x = np.full((n_faces, n_max), RGFGRID_FILL, dtype=np.float64)
    bounds_y = np.full((n_faces, n_max), RGFGRID_FILL, dtype=np.float64)
    for i in range(n_faces):
        for j in range(n_max):
            node_idx = face_nodes[i, j]
            if node_idx != face_fill:
                vi = node_idx - index_offset
                bounds_x[i, j] = node_x[vi]
                bounds_y[i, j] = node_y[vi]
    return bounds_x, bounds_y


def _rgfgrid_compute_edge_faces(face_nodes, edge_nodes, face_fill=RGFGRID_FILL):
    edge_to_faces = defaultdict(list)
    n_faces = face_nodes.shape[0]
    for fi in range(n_faces):
        face = face_nodes[fi]
        valid_nodes = face[face != face_fill]
        nv = len(valid_nodes)
        for j in range(nv):
            n1 = int(valid_nodes[j])
            n2 = int(valid_nodes[(j + 1) % nv])
            key = (min(n1, n2), max(n1, n2))
            face_idx = fi + 1
            if face_idx not in edge_to_faces[key]:
                edge_to_faces[key].append(face_idx)

    edge_faces = np.zeros((len(edge_nodes), 2), dtype=np.int32)
    for ei, (n1, n2) in enumerate(edge_nodes):
        key = (min(int(n1), int(n2)), max(int(n1), int(n2)))
        adjacent = edge_to_faces.get(key, [])
        if len(adjacent) >= 1:
            edge_faces[ei, 0] = adjacent[0]
        if len(adjacent) >= 2:
            edge_faces[ei, 1] = adjacent[1]
    return edge_faces


class ImportNetCDF(Operator, ImportHelper):
    """Import a NetCDF file as a mesh"""
    bl_idname = "import_mesh.netcdf"
    bl_label = "Import NetCDF Mesh"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".nc"
    filter_glob: StringProperty(
        default="*.nc",
        options={'HIDDEN'},
        maxlen=255,
    )

    use_ugrid_convention: BoolProperty(
        name="Use UGRID Convention",
        description="Assume file follows UGRID conventions for unstructured grids",
        default=True,
    )

    auto_scale: BoolProperty(
        name="Auto Scale Coordinates",
        description="Automatically scale coordinates if they appear to be in different units (e.g., degrees to meters)",
        default=True,
    )

    coordinate_scale: FloatProperty(
        name="Coordinate Scale Factor",
        description="Manual scale factor to apply to coordinates (1.0 = no scaling)",
        default=1.0,
        min=0.0001,
        max=1000000.0,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "use_ugrid_convention")
        layout.prop(self, "auto_scale")
        layout.prop(self, "coordinate_scale")

    def execute(self, context):
        if not XARRAY_AVAILABLE:
            self.report({'ERROR'}, "xarray module is not available. Please install it in Blender's Python.")
            return {'CANCELLED'}

        filepath = self.filepath
        if not os.path.exists(filepath):
            self.report({'ERROR'}, f"File not found: {filepath}")
            return {'CANCELLED'}

        try:
            # Open the NetCDF file
            dataset = xr.open_dataset(filepath)
            
            # Try to extract mesh data
            mesh_data = self.extract_mesh_data(dataset)
            
            if mesh_data is None:
                # Provide more detailed error message
                var_info = []
                for var_name in dataset.variables:
                    var = dataset.variables[var_name]
                    var_info.append(f"{var_name}({var.dims})")
                var_list = ", ".join(var_info[:10])  # Limit to first 10
                if len(var_info) > 10:
                    var_list += f", ... ({len(var_info)} total)"
                
                self.report({'ERROR'}, 
                    f"Could not extract mesh data. Found variables: {var_list}. "
                    "Check Blender console for details.")
                return {'CANCELLED'}

            # Create Blender mesh
            mesh_name = os.path.splitext(os.path.basename(filepath))[0]
            self.create_blender_mesh(mesh_data, mesh_name)

            dataset.close()
            
            self.report({'INFO'}, f"Successfully imported {mesh_name}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Error importing NetCDF file: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

    def extract_mesh_data(self, dataset):
        """
        Extract mesh data from NetCDF dataset.
        Supports UGRID conventions (including Delft3D/RFGRID) and generic vertex/edge arrays.
        """
        mesh_data = {
            'vertices': None,
            'edges': None,
            'faces': None,
            'face_fill_value': -1,  # Default fill value
        }

        # Debug: Print available variables for troubleshooting
        debug_vars = []
        for var_name in dataset.variables:
            var = dataset.variables[var_name]
            debug_vars.append(f"{var_name}: dims={var.dims}, shape={var.shape}")
        print(f"Available variables: {debug_vars}")

        # Step 1: Look for UGRID topology variable
        # UGRID uses a topology variable with attributes pointing to coordinate/connectivity vars
        topology_var = None
        node_x_var = None
        node_y_var = None
        node_z_var = None
        face_connectivity_var = None
        edge_connectivity_var = None
        
        if self.use_ugrid_convention:
            # Find topology variable (has topology_dimension attribute)
            for var_name in dataset.variables:
                var = dataset.variables[var_name]
                attrs = getattr(var, 'attrs', {})
                
                # Check for topology variable
                if 'topology_dimension' in attrs or 'cf_role' in attrs:
                    topology_var = var_name
                    topo_attrs = attrs
                    
                    # Extract variable names from topology attributes
                    if 'node_coordinates' in topo_attrs:
                        node_coords_str = topo_attrs['node_coordinates']
                        # Can be space-separated list of variable names
                        node_coord_vars = node_coords_str.split()
                        if len(node_coord_vars) >= 1:
                            node_x_var = node_coord_vars[0]
                        if len(node_coord_vars) >= 2:
                            node_y_var = node_coord_vars[1]
                        if len(node_coord_vars) >= 3:
                            node_z_var = node_coord_vars[2]
                    
                    if 'face_node_connectivity' in topo_attrs:
                        face_connectivity_var = topo_attrs['face_node_connectivity']
                    
                    if 'edge_node_connectivity' in topo_attrs:
                        edge_connectivity_var = topo_attrs['edge_node_connectivity']
                    
                    break

        # Step 2: Extract node coordinates
        # Try multiple strategies
        coord_vars = {'x': None, 'y': None, 'z': None}
        
        # Strategy 1: Use topology variable references
        if node_x_var and node_x_var in dataset.variables:
            coord_vars['x'] = dataset.variables[node_x_var]
        if node_y_var and node_y_var in dataset.variables:
            coord_vars['y'] = dataset.variables[node_y_var]
        if node_z_var and node_z_var in dataset.variables:
            coord_vars['z'] = dataset.variables[node_z_var]
        
        # Strategy 2: Search for variables with 'node' dimension
        if coord_vars['x'] is None or coord_vars['y'] is None:
            # First pass: collect all potential coordinate variables
            potential_coords = {'x': [], 'y': [], 'z': []}
            
            for var_name in dataset.variables:
                var = dataset.variables[var_name]
                dims = var.dims
                
                # Check if this variable has a node dimension
                has_node_dim = False
                node_dim_name = None
                for dim in dims:
                    if 'node' in dim.lower() or dim.lower() in ['nnode', 'n_nodes']:
                        has_node_dim = True
                        node_dim_name = dim
                        break
                
                if has_node_dim and len(dims) == 1:
                    var_lower = var_name.lower()
                    # Check for X coordinate patterns
                    x_patterns = ['x', 'lon', 'longitude', 'node_x', 'mesh2d_node_x', 'mesh_node_x']
                    if any(pattern in var_lower for pattern in x_patterns):
                        potential_coords['x'].append((var_name, var, node_dim_name))
                    # Check for Y coordinate patterns
                    y_patterns = ['y', 'lat', 'latitude', 'node_y', 'mesh2d_node_y', 'mesh_node_y']
                    if any(pattern in var_lower for pattern in y_patterns):
                        potential_coords['y'].append((var_name, var, node_dim_name))
                    # Check for Z coordinate patterns
                    z_patterns = ['z', 'depth', 'elevation', 'node_z', 'mesh2d_node_z', 'mesh_node_z']
                    if any(pattern in var_lower for pattern in z_patterns):
                        potential_coords['z'].append((var_name, var, node_dim_name))
            
            # Select coordinates that share the same dimension (most common case)
            # If we have multiple candidates, prefer ones with matching dimensions
            if potential_coords['x'] and potential_coords['y']:
                # Try to find X and Y with matching dimensions
                for x_name, x_var, x_dim in potential_coords['x']:
                    for y_name, y_var, y_dim in potential_coords['y']:
                        if x_dim == y_dim:
                            coord_vars['x'] = x_var
                            coord_vars['y'] = y_var
                            print(f"Matched coordinates: {x_name} and {y_name} (dim: {x_dim})")
                            break
                    if coord_vars['x'] is not None:
                        break
                
                # If no match, just use the first ones
                if coord_vars['x'] is None:
                    coord_vars['x'] = potential_coords['x'][0][1]
                    coord_vars['y'] = potential_coords['y'][0][1]
                    print(f"Using first available: {potential_coords['x'][0][0]} and {potential_coords['y'][0][0]}")
            
            # Handle Z coordinate
            if potential_coords['z']:
                # Try to match with X/Y dimension
                if coord_vars['x'] is not None:
                    x_dim = coord_vars['x'].dims[0] if coord_vars['x'].dims else None
                    for z_name, z_var, z_dim in potential_coords['z']:
                        if x_dim and z_dim == x_dim:
                            coord_vars['z'] = z_var
                            break
                # If no match, use first available
                if coord_vars['z'] is None:
                    coord_vars['z'] = potential_coords['z'][0][1]
        
        # Strategy 3: Look for 2D coordinate arrays
        if coord_vars['x'] is None or coord_vars['y'] is None:
            for var_name in dataset.variables:
                var = dataset.variables[var_name]
                dims = var.dims
                
                # Check for 2D coordinate array (node, 2) or (node, 3)
                if len(dims) == 2:
                    dim0, dim1 = dims
                    if ('node' in dim0.lower() or 'nnode' in dim0.lower()) and dim1 in [2, 3]:
                        # This might be a combined coordinate array
                        coords_array = var.values
                        if coords_array.shape[1] >= 2:
                            # Create temporary coordinate variables
                            if coord_vars['x'] is None:
                                coord_vars['x'] = type('obj', (object,), {'values': coords_array[:, 0]})()
                            if coord_vars['y'] is None:
                                coord_vars['y'] = type('obj', (object,), {'values': coords_array[:, 1]})()
                            if coords_array.shape[1] >= 3 and coord_vars['z'] is None:
                                coord_vars['z'] = type('obj', (object,), {'values': coords_array[:, 2]})()

        # Build vertices array
        if coord_vars['x'] is not None and coord_vars['y'] is not None:
            x_vals = coord_vars['x'].values if hasattr(coord_vars['x'], 'values') else coord_vars['x']
            y_vals = coord_vars['y'].values if hasattr(coord_vars['y'], 'values') else coord_vars['y']
            
            n_nodes = len(x_vals)
            vertices = np.zeros((n_nodes, 3))
            vertices[:, 0] = x_vals
            vertices[:, 1] = y_vals
            
            if coord_vars['z'] is not None:
                z_vals = coord_vars['z'].values if hasattr(coord_vars['z'], 'values') else coord_vars['z']
                vertices[:, 2] = z_vals
            else:
                vertices[:, 2] = 0.0
            
            # Apply coordinate scaling if needed
            if self.coordinate_scale != 1.0:
                vertices = vertices * self.coordinate_scale
                print(f"Applied coordinate scale factor: {self.coordinate_scale}")
            
            # Auto-detect and scale if coordinates appear to be in degrees (geographic)
            if self.auto_scale:
                x_range = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
                y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
                
                # If coordinates are in typical degree ranges (-180 to 180, -90 to 90)
                # and the range is small, they might need scaling
                if (abs(np.max(vertices[:, 0])) < 200 and abs(np.max(vertices[:, 1])) < 200 and
                    x_range < 10 and y_range < 10 and x_range > 0.001):
                    # Likely degrees, but small range - might be fine as-is
                    # Check if mesh appears too small (less than 0.1 Blender units)
                    if x_range < 0.1 or y_range < 0.1:
                        # Scale up - assume coordinates are in meters but very small
                        # or in degrees and need conversion
                        scale_factor = 1.0 / max(x_range, y_range) * 10.0  # Scale to ~10 units
                        vertices = vertices * scale_factor
                        print(f"Auto-scaled coordinates (detected small range): factor={scale_factor:.2f}")
            
            mesh_data['vertices'] = vertices
            print(f"Found {n_nodes} vertices")

        # Step 3: Extract face connectivity
        # Strategy 1: Use topology variable reference
        if face_connectivity_var and face_connectivity_var in dataset.variables:
            face_var = dataset.variables[face_connectivity_var]
            mesh_data['faces'] = face_var.values
            # Check for fill value attribute
            if hasattr(face_var, 'attrs') and '_FillValue' in face_var.attrs:
                mesh_data['face_fill_value'] = face_var.attrs['_FillValue']
            print(f"Found face connectivity: {face_connectivity_var}")
        
        # Strategy 2: Search by CF role
        if mesh_data['faces'] is None:
            for var_name in dataset.variables:
                var = dataset.variables[var_name]
                attrs = getattr(var, 'attrs', {})
                
                if 'cf_role' in attrs and attrs['cf_role'] == 'face_node_connectivity':
                    mesh_data['faces'] = var.values
                    if '_FillValue' in attrs:
                        mesh_data['face_fill_value'] = attrs['_FillValue']
                    print(f"Found face connectivity by CF role: {var_name}")
                    break
        
        # Strategy 3: Search by name patterns
        if mesh_data['faces'] is None:
            for var_name in dataset.variables:
                var = dataset.variables[var_name]
                var_lower = var_name.lower()
                
                if ('face' in var_lower and 'node' in var_lower) or \
                   ('face' in var_lower and 'connect' in var_lower) or \
                   'face_node_connectivity' in var_lower or \
                   'mesh2d_face_nodes' in var_lower:
                    # Check dimensions - should be (face, max_nodes_per_face)
                    if len(var.dims) == 2:
                        mesh_data['faces'] = var.values
                        attrs = getattr(var, 'attrs', {})
                        if '_FillValue' in attrs:
                            mesh_data['face_fill_value'] = attrs['_FillValue']
                        print(f"Found face connectivity by name: {var_name}")
                        break

        # Step 4: Extract edge connectivity
        # Strategy 1: Use topology variable reference
        if edge_connectivity_var and edge_connectivity_var in dataset.variables:
            mesh_data['edges'] = dataset.variables[edge_connectivity_var].values
            print(f"Found edge connectivity: {edge_connectivity_var}")
        
        # Strategy 2: Search by CF role
        if mesh_data['edges'] is None:
            for var_name in dataset.variables:
                var = dataset.variables[var_name]
                attrs = getattr(var, 'attrs', {})
                
                if 'cf_role' in attrs and attrs['cf_role'] == 'edge_node_connectivity':
                    mesh_data['edges'] = var.values
                    print(f"Found edge connectivity by CF role: {var_name}")
                    break
        
        # Strategy 3: Search by name patterns
        if mesh_data['edges'] is None:
            for var_name in dataset.variables:
                var = dataset.variables[var_name]
                var_lower = var_name.lower()
                
                if ('edge' in var_lower and 'node' in var_lower) or \
                   ('edge' in var_lower and 'connect' in var_lower) or \
                   'edge_node_connectivity' in var_lower or \
                   'mesh2d_edge_nodes' in var_lower:
                    # Check dimensions - should be (edge, 2)
                    if len(var.dims) == 2 and var.shape[1] == 2:
                        mesh_data['edges'] = var.values
                        print(f"Found edge connectivity by name: {var_name}")
                        break

        # Fallback: try generic variable names
        if mesh_data['vertices'] is None:
            for var_name in ['vertices', 'nodes', 'points', 'coordinates']:
                if var_name in dataset.variables:
                    var = dataset.variables[var_name]
                    if len(var.dims) == 2 and var.shape[1] >= 2:
                        vertices = var.values
                        if vertices.shape[1] == 2:
                            z_coord = np.zeros((vertices.shape[0], 1))
                            vertices = np.hstack([vertices, z_coord])
                        mesh_data['vertices'] = vertices
                        print(f"Found vertices by generic name: {var_name}")
                        break

        # If we have vertices, return the mesh data
        if mesh_data['vertices'] is not None:
            return mesh_data
        
        # Print diagnostic information
        print("Failed to extract mesh data. Available variables:")
        for var_name in dataset.variables:
            var = dataset.variables[var_name]
            print(f"  {var_name}: dims={var.dims}, shape={var.shape}, attrs={getattr(var, 'attrs', {})}")
        
        return None

    def create_blender_mesh(self, mesh_data, mesh_name):
        """Create a Blender mesh object from extracted mesh data with validation"""
        try:
            vertices = mesh_data['vertices']
            edges = mesh_data['edges']
            faces = mesh_data['faces']

            if vertices is None or len(vertices) == 0:
                print(f"Error: No vertices found for {mesh_name}")
                return

            # Validate and convert vertices
            num_verts = len(vertices)
            print(f"Creating mesh with {num_verts} vertices")
            
            # Check for coordinate issues (NaN, Inf, or extreme values)
            vertices_array = np.array(vertices)
            if np.any(np.isnan(vertices_array)) or np.any(np.isinf(vertices_array)):
                print("Warning: Found NaN or Inf values in coordinates, replacing with 0")
                vertices_array = np.nan_to_num(vertices_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check coordinate ranges
            coord_ranges = {
                'x': (np.min(vertices_array[:, 0]), np.max(vertices_array[:, 0])),
                'y': (np.min(vertices_array[:, 1]), np.max(vertices_array[:, 1])),
                'z': (np.min(vertices_array[:, 2]), np.max(vertices_array[:, 2]))
            }
            print(f"Coordinate ranges: X={coord_ranges['x']}, Y={coord_ranges['y']}, Z={coord_ranges['z']}")
            
            # Check for vertices at origin (potential issue)
            origin_threshold = 1e-6
            at_origin = np.sum(np.abs(vertices_array) < origin_threshold, axis=1)
            vertices_at_origin = np.sum(at_origin == 3)  # All three coordinates near zero
            if vertices_at_origin > 0:
                print(f"Warning: {vertices_at_origin} vertices are at or near origin (0,0,0)")
                if vertices_at_origin > num_verts * 0.1:  # More than 10% at origin
                    print("Warning: Many vertices at origin - this may indicate an indexing or coordinate extraction problem")
            
            # Convert vertices to list of tuples
            verts = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices_array]
            
            # Detect if indices are 0-based or 1-based
            # Check connectivity arrays to determine indexing
            index_offset = 0
            if faces is not None:
                faces_array = np.array(faces)
                min_index = np.min(faces_array[faces_array >= 0])
                if min_index > 0 and min_index < num_verts:
                    # Likely 1-based indexing
                    index_offset = 1
                    print(f"Detected 1-based indexing (min index: {min_index}), converting to 0-based")
            
            if edges is not None:
                edges_array = np.array(edges)
                min_edge_index = np.min(edges_array[edges_array >= 0])
                if min_edge_index > 0 and min_edge_index < num_verts:
                    index_offset = max(index_offset, 1)
            
            # Process edges with validation
            edge_list = []
            if edges is not None:
                edges_array = np.array(edges, dtype=np.int32)
                if edges_array.ndim == 2 and edges_array.shape[1] == 2:
                    for e in edges_array:
                        v0 = int(e[0]) - index_offset
                        v1 = int(e[1]) - index_offset
                        # Validate indices
                        if 0 <= v0 < num_verts and 0 <= v1 < num_verts and v0 != v1:
                            edge_list.append((v0, v1))
                        else:
                            if v0 < 0 or v0 >= num_verts or v1 < 0 or v1 >= num_verts:
                                print(f"Warning: Invalid edge index ({e[0]}, {e[1]}) -> ({v0}, {v1}), skipping")
                    print(f"Created {len(edge_list)} valid edges")
            
            # Process faces with validation
            face_list = []
            if faces is not None:
                faces_array = np.array(faces, dtype=np.int32)
                if faces_array.ndim == 2:
                    max_valid_index = num_verts - 1
                    invalid_faces = 0
                    fill_value = mesh_data.get('face_fill_value', -1)
                    
                    for i, face in enumerate(faces_array):
                        # Remove fill values and convert indices
                        face_verts = []
                        for v in face:
                            # Check for fill value first (before offset)
                            if v == fill_value or (fill_value < 0 and v < 0):
                                # Fill value detected, stop processing this face
                                break
                            
                            v_int = int(v) - index_offset
                            
                            # Check for fill values (negative after offset, or out of range)
                            if v_int < 0:
                                # Invalid index, skip
                                continue
                            elif v_int > max_valid_index:
                                # Out of range, skip
                                continue
                            elif v_int not in face_verts:  # Avoid duplicate vertices
                                face_verts.append(v_int)
                        
                        # Validate face
                        if len(face_verts) >= 3:
                            # Remove duplicates while preserving order
                            seen = set()
                            unique_face_verts = []
                            for v in face_verts:
                                if v not in seen:
                                    seen.add(v)
                                    unique_face_verts.append(v)
                            
                            # Check for degenerate faces (need at least 3 unique vertices)
                            if len(unique_face_verts) >= 3:
                                # Additional check: ensure vertices form a valid polygon
                                # (not all collinear - simplified check)
                                if len(unique_face_verts) == 3:
                                    # Triangle - always valid if 3 unique vertices
                                    face_list.append(tuple(unique_face_verts))
                                elif len(unique_face_verts) == 4:
                                    # Quad - check if it's not degenerate
                                    face_list.append(tuple(unique_face_verts))
                                else:
                                    # Polygon - add as-is (Blender will handle it)
                                    face_list.append(tuple(unique_face_verts))
                            else:
                                invalid_faces += 1
                        else:
                            invalid_faces += 1
                    
                    if invalid_faces > 0:
                        print(f"Warning: Skipped {invalid_faces} invalid faces")
                    print(f"Created {len(face_list)} valid faces")
            
            # Validate mesh data before creation
            if len(verts) == 0:
                print(f"Error: No valid vertices for {mesh_name}")
                return
            
            # Create new mesh
            mesh = bpy.data.meshes.new(name=mesh_name)
            
            # Create mesh from validated data
            try:
                mesh.from_pydata(verts, edge_list, face_list)
                mesh.update()
                
                # Additional validation
                if len(mesh.vertices) == 0:
                    print(f"Error: Mesh creation resulted in 0 vertices for {mesh_name}")
                    bpy.data.meshes.remove(mesh)
                    return
                
                # Check for invalid geometry
                mesh.validate(verbose=True)
                
            except Exception as e:
                print(f"Error creating mesh geometry: {e}")
                import traceback
                traceback.print_exc()
                bpy.data.meshes.remove(mesh)
                return

            # Create object and link to scene
            obj = bpy.data.objects.new(mesh_name, mesh)
            bpy.context.collection.objects.link(obj)
            
            # Select and make active
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            print(f"Successfully created mesh object: {mesh_name} with {len(mesh.vertices)} vertices, {len(mesh.edges)} edges, {len(mesh.polygons)} faces")
            
        except Exception as e:
            print(f"Error in create_blender_mesh: {e}")
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Error creating mesh: {str(e)}")


class ExportNetCDF(Operator, ExportHelper):
    """Export a Blender mesh to NetCDF file"""
    bl_idname = "export_mesh.netcdf"
    bl_label = "Export NetCDF Mesh"
    bl_options = {'PRESET'}

    filename_ext = ".nc"
    filter_glob: StringProperty(
        default="*.nc",
        options={'HIDDEN'},
        maxlen=255,
    )

    use_ugrid_convention: BoolProperty(
        name="Use UGRID Convention",
        description="Export following UGRID conventions for unstructured grids",
        default=True,
    )

    use_1_based_indexing: BoolProperty(
        name="Use 1-Based Indexing",
        description="Use 1-based indexing for connectivity arrays (standard for UGRID)",
        default=True,
    )

    include_edges: BoolProperty(
        name="Include Edges",
        description="Export edge connectivity information",
        default=True,
    )

    use_delft3d_format: BoolProperty(
        name="Delft3D/RFGRID Format",
        description="Use Delft3D/RFGRID naming conventions (mesh2d_* variables)",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "use_ugrid_convention")
        layout.prop(self, "use_delft3d_format")
        layout.prop(self, "use_1_based_indexing")
        layout.prop(self, "include_edges")

    def execute(self, context):
        if not XARRAY_AVAILABLE:
            self.report({'ERROR'}, "xarray module is not available. Please install it in Blender's Python.")
            return {'CANCELLED'}

        # Get the active object
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object selected")
            return {'CANCELLED'}

        mesh = obj.data
        if len(mesh.vertices) == 0:
            self.report({'ERROR'}, "Mesh has no vertices")
            return {'CANCELLED'}

        try:
            # Extract mesh data
            mesh_data = self.extract_blender_mesh_data(mesh)
            
            # Create NetCDF dataset
            self.create_netcdf_file(mesh_data, self.filepath, obj.name)
            
            self.report({'INFO'}, f"Successfully exported {obj.name} to {self.filepath}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Error exporting NetCDF file: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

    def extract_blender_mesh_data(self, mesh):
        """Extract mesh data from Blender mesh"""
        # Get vertices (world coordinates)
        vertices = np.array([v.co for v in mesh.vertices])
        
        # Get edges
        edges = None
        if len(mesh.edges) > 0:
            edges = np.array([e.vertices for e in mesh.edges])
        
        # Get faces (polygons)
        faces = None
        if len(mesh.polygons) > 0:
            # Collect all face vertex indices
            face_list = []
            for poly in mesh.polygons:
                face_list.append([v for v in poly.vertices])
            
            # Find maximum vertices per face
            max_verts_per_face = max(len(f) for f in face_list) if face_list else 3
            
            # Pad faces to same length with fill value
            fill_value = -1
            padded_faces = []
            for face in face_list:
                padded = list(face)
                while len(padded) < max_verts_per_face:
                    padded.append(fill_value)
                padded_faces.append(padded)
            
            faces = np.array(padded_faces, dtype=np.int32)
        
        return {
            'vertices': vertices,
            'edges': edges,
            'faces': faces,
            'fill_value': -1,
        }

    def _create_rgfgrid_netcdf(self, mesh_data, filepath, mesh_name):
        """Create a NetCDF file matching RGFGRID/Delft3D conventions."""
        vertices = mesh_data['vertices']
        edges = mesh_data['edges']
        faces = mesh_data['faces']
        fill_value = RGFGRID_FILL

        num_nodes = len(vertices)
        num_faces = len(faces) if faces is not None else 0
        num_edges = len(edges) if edges is not None else 0
        index_offset = 1 if self.use_1_based_indexing else 0

        prefix = 'mesh2d'
        node_dim = f'{prefix}_nNodes'
        face_dim = f'{prefix}_nFaces'
        edge_dim = f'{prefix}_nEdges'
        nmax_face_dim = f'{prefix}_nMax_face_nodes'
        two_dim = 'Two'

        node_x = vertices[:, 0].astype(np.float64)
        node_y = vertices[:, 1].astype(np.float64)
        node_z = np.full(num_nodes, np.nan, dtype=np.float64)

        data_vars = {}

        data_vars[f'{prefix}_node_x'] = ([node_dim], node_x, {
            'units': 'm',
            'standard_name': 'projection_x_coordinate',
            'long_name': 'x-coordinate of mesh nodes',
        })
        data_vars[f'{prefix}_node_y'] = ([node_dim], node_y, {
            'units': 'm',
            'standard_name': 'projection_y_coordinate',
            'long_name': 'y-coordinate of mesh nodes',
        })
        data_vars[f'{prefix}_node_z'] = ([node_dim], node_z, {
            'mesh': prefix,
            'location': 'node',
            'coordinates': f'{prefix}_node_x {prefix}_node_y',
            'standard_name': 'altitude',
            'long_name': 'z-coordinate of mesh nodes',
            'units': 'm',
            'grid_mapping': 'projected_coordinate_system',
        })

        face_connectivity = None
        if faces is not None and num_faces > 0:
            face_connectivity = faces.copy().astype(np.int32)
            if index_offset == 1:
                face_connectivity = np.where(
                    face_connectivity >= 0,
                    face_connectivity + index_offset,
                    fill_value,
                ).astype(np.int32)
            face_connectivity = np.where(
                face_connectivity < 0, fill_value, face_connectivity
            ).astype(np.int32)

            data_vars[f'{prefix}_face_nodes'] = (
                [face_dim, nmax_face_dim], face_connectivity, {
                    'cf_role': 'face_node_connectivity',
                    'long_name': 'Vertex nodes of mesh faces (counterclockwise)',
                    'start_index': np.int32(index_offset),
                })

            face_x, face_y = _rgfgrid_compute_face_centroids(
                node_x, node_y, face_connectivity, fill_value, index_offset)
            face_x_bnd, face_y_bnd = _rgfgrid_compute_face_bounds(
                node_x, node_y, face_connectivity, fill_value, index_offset)

            data_vars[f'{prefix}_face_x'] = ([face_dim], face_x, {
                'units': 'm',
                'standard_name': 'projection_x_coordinate',
                'long_name': 'Characteristic x-coordinate of mesh face',
                'bounds': f'{prefix}_face_x_bnd',
            })
            data_vars[f'{prefix}_face_y'] = ([face_dim], face_y, {
                'units': 'm',
                'standard_name': 'projection_y_coordinate',
                'long_name': 'Characteristic y-coordinate of mesh face',
                'bounds': f'{prefix}_face_y_bnd',
            })
            data_vars[f'{prefix}_face_x_bnd'] = (
                [face_dim, nmax_face_dim], face_x_bnd, {
                    'units': 'm',
                    'standard_name': 'projection_x_coordinate',
                    'long_name': 'x-coordinate bounds of mesh faces (i.e. corner coordinates)',
                })
            data_vars[f'{prefix}_face_y_bnd'] = (
                [face_dim, nmax_face_dim], face_y_bnd, {
                    'units': 'm',
                    'standard_name': 'projection_y_coordinate',
                    'long_name': 'y-coordinate bounds of mesh faces (i.e. corner coordinates)',
                })

        edge_connectivity = None
        if edges is not None and num_edges > 0 and self.include_edges:
            edge_connectivity = edges.copy().astype(np.int32)
            if index_offset == 1:
                edge_connectivity = edge_connectivity + index_offset

            data_vars[f'{prefix}_edge_nodes'] = (
                [edge_dim, two_dim], edge_connectivity, {
                    'cf_role': 'edge_node_connectivity',
                    'long_name': 'Start and end nodes of mesh edges',
                    'start_index': np.int32(index_offset),
                })

            edge_x, edge_y = _rgfgrid_compute_edge_midpoints(
                node_x, node_y, edge_connectivity, index_offset)
            data_vars[f'{prefix}_edge_x'] = ([edge_dim], edge_x, {
                'units': 'm',
                'standard_name': 'projection_x_coordinate',
                'long_name': 'characteristic x-coordinate of the mesh edge (e.g. midpoint)',
            })
            data_vars[f'{prefix}_edge_y'] = ([edge_dim], edge_y, {
                'units': 'm',
                'standard_name': 'projection_y_coordinate',
                'long_name': 'characteristic y-coordinate of the mesh edge (e.g. midpoint)',
            })

            if face_connectivity is not None:
                edge_faces = _rgfgrid_compute_edge_faces(face_connectivity, edge_connectivity)
                data_vars[f'{prefix}_edge_faces'] = (
                    [edge_dim, two_dim], edge_faces, {
                        'cf_role': 'edge_face_connectivity',
                        'long_name': 'Neighboring faces of mesh edges',
                        'start_index': np.int32(index_offset),
                    })

        if self.use_ugrid_convention:
            topology_attrs = {
                'cf_role': 'mesh_topology',
                'long_name': 'Topology data of 2D mesh',
                'topology_dimension': np.int32(2 if num_faces > 0 else 1),
                'node_coordinates': f'{prefix}_node_x {prefix}_node_y',
                'node_dimension': node_dim,
                'max_face_nodes_dimension': nmax_face_dim,
            }
            if edge_connectivity is not None and self.include_edges:
                topology_attrs['edge_node_connectivity'] = f'{prefix}_edge_nodes'
                topology_attrs['edge_dimension'] = edge_dim
                topology_attrs['edge_coordinates'] = f'{prefix}_edge_x {prefix}_edge_y'
                topology_attrs['edge_face_connectivity'] = f'{prefix}_edge_faces'
            if face_connectivity is not None:
                topology_attrs['face_node_connectivity'] = f'{prefix}_face_nodes'
                topology_attrs['face_dimension'] = face_dim
                topology_attrs['face_coordinates'] = f'{prefix}_face_x {prefix}_face_y'

            data_vars[prefix] = ([], RGFGRID_TOPOLOGY_VALUE, topology_attrs)

        data_vars['projected_coordinate_system'] = ([], RGFGRID_TOPOLOGY_VALUE, {
            'name': 'Unknown projected',
            'epsg': np.int32(0),
            'grid_mapping_name': 'Unknown projected',
            'longitude_of_prime_meridian': np.float64(0.0),
            'semi_major_axis': np.float64(6378137.0),
            'semi_minor_axis': np.float64(6356752.314245),
            'inverse_flattening': np.float64(298.257223563),
            'EPSG_code': '',
            'value': 'value is equal to EPSG code',
            'proj_string': '',
        })

        global_attrs = {
            'institution': 'Deltares',
            'references': 'http://www.deltares.nl',
            'source': f'Blender NetCDF Mesh Exporter. Model: {mesh_name}',
            'history': f'Created by Blender NetCDF exporter from {mesh_name}',
            'Conventions': 'CF-1.8 UGRID-1.0 Deltares-0.10',
        }

        ds = xr.Dataset(data_vars, attrs=global_attrs)

        encoding = {}
        float_vars = {
            f'{prefix}_node_x', f'{prefix}_node_y', f'{prefix}_node_z',
            f'{prefix}_edge_x', f'{prefix}_edge_y',
            f'{prefix}_face_x', f'{prefix}_face_y',
            f'{prefix}_face_x_bnd', f'{prefix}_face_y_bnd',
        }
        int_fill_vars = {
            f'{prefix}_face_nodes': np.int32(fill_value),
            f'{prefix}_edge_faces': np.int32(fill_value),
        }
        float_fill_vars = {
            f'{prefix}_node_z': np.float64(fill_value),
            f'{prefix}_face_x_bnd': np.float64(fill_value),
            f'{prefix}_face_y_bnd': np.float64(fill_value),
        }

        for var_name in ds.data_vars:
            var_encoding = {}
            if var_name in float_vars:
                var_encoding['dtype'] = 'float64'
            elif var_name in (prefix, 'projected_coordinate_system'):
                var_encoding['dtype'] = 'int32'
            elif var_name in (f'{prefix}_face_nodes', f'{prefix}_edge_nodes',
                              f'{prefix}_edge_faces'):
                var_encoding['dtype'] = 'int32'

            if var_name in int_fill_vars:
                var_encoding['_FillValue'] = int_fill_vars[var_name]
            elif var_name in float_fill_vars:
                var_encoding['_FillValue'] = float_fill_vars[var_name]

            if var_encoding:
                encoding[var_name] = var_encoding

        ds.to_netcdf(filepath, encoding=encoding, format='NETCDF3_CLASSIC')
        ds.close()

        print(f"Exported RGFGRID mesh to {filepath}")
        print(f"  Nodes: {num_nodes}, Faces: {num_faces}, Edges: {num_edges}")

    def create_netcdf_file(self, mesh_data, filepath, mesh_name):
        """Create a NetCDF file from mesh data following UGRID conventions"""
        if self.use_delft3d_format:
            self._create_rgfgrid_netcdf(mesh_data, filepath, mesh_name)
            return

        vertices = mesh_data['vertices']
        edges = mesh_data['edges']
        faces = mesh_data['faces']
        fill_value = mesh_data['fill_value']
        
        num_nodes = len(vertices)
        num_faces = len(faces) if faces is not None else 0
        num_edges = len(edges) if edges is not None else 0
        
        # Determine index offset
        index_offset = 1 if self.use_1_based_indexing else 0
        
        # Choose variable naming convention
        if self.use_delft3d_format:
            # Delft3D/RFGRID naming convention
            prefix = 'mesh2d'
            node_x_name = f'{prefix}_node_x'
            node_y_name = f'{prefix}_node_y'
            node_z_name = f'{prefix}_node_z'
            face_conn_name = f'{prefix}_face_nodes'
            edge_conn_name = f'{prefix}_edge_nodes'
            topology_name = prefix
            node_dim = f'{prefix}_nNodes'
            face_dim = f'{prefix}_nFaces'
            edge_dim = f'{prefix}_nEdges'
            nmax_face_dim = f'{prefix}_nMaxFaceNodes'
        else:
            # Standard UGRID naming convention
            prefix = ''
            node_x_name = 'node_x'
            node_y_name = 'node_y'
            node_z_name = 'node_z'
            face_conn_name = 'face_node_connectivity'
            edge_conn_name = 'edge_node_connectivity'
            topology_name = 'mesh_topology'
            node_dim = 'node'
            face_dim = 'face'
            edge_dim = 'edge'
            nmax_face_dim = 'nmax_face'
        
        # Node coordinates
        node_x = vertices[:, 0]
        node_y = vertices[:, 1]
        node_z = vertices[:, 2] if vertices.shape[1] > 2 else np.zeros(num_nodes)
        
        # Create coordinate variables as data arrays (not coords in xarray)
        data_vars = {}
        
        data_vars[node_x_name] = ([node_dim], node_x, {
            'long_name': 'x-coordinate of mesh nodes',
            'standard_name': 'projection_x_coordinate',
            'units': 'm',
        })
        
        data_vars[node_y_name] = ([node_dim], node_y, {
            'long_name': 'y-coordinate of mesh nodes',
            'standard_name': 'projection_y_coordinate',
            'units': 'm',
        })
        
        data_vars[node_z_name] = ([node_dim], node_z, {
            'long_name': 'z-coordinate of mesh nodes',
            'standard_name': 'projection_z_coordinate',
            'units': 'm',
        })
        
        # Face node connectivity
        if faces is not None and num_faces > 0:
            # Convert to 1-based indexing if needed
            face_connectivity = faces.copy().astype(np.int32)
            if index_offset == 1:
                # Add offset, but keep fill values as -1
                face_connectivity = np.where(face_connectivity >= 0, 
                                            face_connectivity + index_offset, 
                                            fill_value).astype(np.int32)
            
            nmax_face = face_connectivity.shape[1]
            
            # Create face connectivity variable with proper fill value handling
            # Use masked array or ensure fill values are properly set
            face_conn_attrs = {
                'long_name': 'Maps every face to its nodes',
                'cf_role': 'face_node_connectivity',
                'start_index': np.int32(index_offset),
            }
            
            # Ensure fill values are actually in the array where needed
            # (they should already be, but double-check)
            face_connectivity_filled = np.where(face_connectivity < 0, fill_value, face_connectivity).astype(np.int32)
            
            data_vars[face_conn_name] = ([face_dim, nmax_face_dim], face_connectivity_filled, face_conn_attrs)
        
        # Edge node connectivity
        if edges is not None and num_edges > 0 and self.include_edges:
            # Convert to 1-based indexing if needed
            edge_connectivity = edges.copy().astype(np.int32)
            if index_offset == 1:
                edge_connectivity = edge_connectivity + index_offset
            
            two_dim = 'two' if not self.use_delft3d_format else f'{prefix}_nMaxEdgeNodes'
            if self.use_delft3d_format and two_dim not in [face_dim, node_dim, edge_dim]:
                # For Delft3D, edges typically use dimension 2
                two_dim = 'two'
            
            data_vars[edge_conn_name] = ([edge_dim, two_dim], edge_connectivity, {
                'long_name': 'Maps every edge to the two nodes that it connects',
                'cf_role': 'edge_node_connectivity',
                'start_index': np.int32(index_offset),
            })
        
        # Create topology variable (UGRID convention)
        if self.use_ugrid_convention:
            if self.use_delft3d_format:
                # Delft3D format: topology variable references use full variable names
                node_coords_str = f'{node_x_name} {node_y_name} {node_z_name}'
                topology_attrs = {
                    'topology_dimension': np.int32(2 if num_faces > 0 else 1),
                    'node_coordinates': node_coords_str,
                    'long_name': 'Topology data of 2D unstructured mesh',
                }
                
                if faces is not None and num_faces > 0:
                    topology_attrs['face_node_connectivity'] = face_conn_name
                    topology_attrs['face_dimension'] = face_dim
                
                if edges is not None and num_edges > 0 and self.include_edges:
                    topology_attrs['edge_node_connectivity'] = edge_conn_name
                    topology_attrs['edge_dimension'] = edge_dim
            else:
                # Standard UGRID format
                topology_attrs = {
                    'topology_dimension': np.int32(2 if num_faces > 0 else 1),
                    'node_coordinates': f'{node_x_name} {node_y_name} {node_z_name}',
                    'long_name': 'Topology data of 2D unstructured mesh',
                }
                
                if faces is not None and num_faces > 0:
                    topology_attrs['face_node_connectivity'] = face_conn_name
                    topology_attrs['face_dimension'] = face_dim
                
                if edges is not None and num_edges > 0 and self.include_edges:
                    topology_attrs['edge_node_connectivity'] = edge_conn_name
                    topology_attrs['edge_dimension'] = edge_dim
            
            # Create a scalar variable to store topology attributes
            data_vars[topology_name] = ([], np.int32(0), topology_attrs)
        
        # Global attributes
        global_attrs = {
            'title': f'Mesh exported from Blender: {mesh_name}',
            'source': 'Blender NetCDF Mesh Exporter',
            'Conventions': 'CF-1.6 UGRID-1.0',
            'institution': 'Blender',
            'history': f'Created by Blender NetCDF exporter',
        }
        
        # Add coordinate reference system for Delft3D compatibility
        if self.use_delft3d_format:
            # Create a CRS variable (coordinate reference system)
            # Delft3D often expects this for proper coordinate handling
            crs_attrs = {
                'grid_mapping_name': 'unknown',
                'long_name': 'coordinate reference system',
            }
            data_vars['crs'] = ([], np.int32(0), crs_attrs)
            
            # Add grid_mapping attribute to coordinate variables
            if node_x_name in data_vars:
                # Update attributes to include grid_mapping reference
                x_attrs = data_vars[node_x_name][2].copy()
                x_attrs['grid_mapping'] = 'crs'
                data_vars[node_x_name] = (data_vars[node_x_name][0], data_vars[node_x_name][1], x_attrs)
            
            if node_y_name in data_vars:
                y_attrs = data_vars[node_y_name][2].copy()
                y_attrs['grid_mapping'] = 'crs'
                data_vars[node_y_name] = (data_vars[node_y_name][0], data_vars[node_y_name][1], y_attrs)
            
            if node_z_name in data_vars:
                z_attrs = data_vars[node_z_name][2].copy()
                z_attrs['grid_mapping'] = 'crs'
                data_vars[node_z_name] = (data_vars[node_z_name][0], data_vars[node_z_name][1], z_attrs)
        
        # Create xarray dataset
        # xarray will automatically create dimensions from the data arrays
        # The dimension names are already set correctly in the data_vars definitions above
        ds = xr.Dataset(data_vars, attrs=global_attrs)
        
        # Verify dimensions are correct (for debugging)
        if self.use_delft3d_format:
            expected_dims = [node_dim]
            if num_faces > 0:
                expected_dims.extend([face_dim, nmax_face_dim])
            if num_edges > 0 and self.include_edges:
                expected_dims.append(edge_dim)
            
            print(f"Created dimensions: {list(ds.dims.keys())}")
            print(f"Expected dimensions: {expected_dims}")
        
        # Write to file
        encoding = {}
        # Set compression for better file size
        # Also set fill values explicitly in encoding
        for var_name in ds.data_vars:
            var_encoding = {}
            if var_name != topology_name and var_name != 'crs':  # Don't compress scalar variables
                var_encoding['zlib'] = True
                var_encoding['complevel'] = 4
            
            if var_encoding:
                encoding[var_name] = var_encoding

        # _FillValue must be in encoding only, not in variable attrs (xarray requirement)
        if faces is not None and num_faces > 0 and face_conn_name in ds.data_vars:
            encoding.setdefault(face_conn_name, {})['_FillValue'] = np.int32(fill_value)
        
        # Write with explicit format (NETCDF4_CLASSIC for better compatibility)
        # Verify the dataset before writing
        print("\n--- Pre-write Verification ---")
        print(f"Variables in dataset: {sorted(ds.data_vars.keys())}")
        print(f"Dimensions: {dict(ds.dims)}")
        
        # Verify topology variable exists and has correct attributes
        if topology_name in ds.variables:
            topo = ds.variables[topology_name]
            print(f"\nTopology variable '{topology_name}':")
            print(f"  Dimensions: {topo.dims}")
            print(f"  Shape: {topo.shape}")
            print(f"  Attributes: {dict(topo.attrs)}")
        
        ds.to_netcdf(filepath, encoding=encoding, format='NETCDF4_CLASSIC')
        ds.close()
        
        # Verify the written file
        print("\n--- Post-write Verification ---")
        verify_ds = xr.open_dataset(filepath)
        print(f"Variables in file: {sorted(verify_ds.variables.keys())}")
        print(f"Dimensions: {dict(verify_ds.dims)}")
        if topology_name in verify_ds.variables:
            verify_topo = verify_ds.variables[topology_name]
            print(f"Topology '{topology_name}' attributes: {dict(verify_topo.attrs)}")
        verify_ds.close()
        
        print(f"Exported mesh to {filepath}")
        print(f"  Format: {'Delft3D/RFGRID' if self.use_delft3d_format else 'Standard UGRID'}")
        print(f"  Nodes: {num_nodes}")
        print(f"  Faces: {num_faces}")
        print(f"  Edges: {num_edges}")
        print(f"  Index offset: {index_offset} (1-based: {self.use_1_based_indexing})")
        if self.use_delft3d_format:
            print(f"  Topology variable: {topology_name}")
            if topology_name in ds.variables:
                topo_var = ds.variables[topology_name]
                print(f"    Attributes: {dict(topo_var.attrs)}")
            print(f"  Node coordinates: {node_x_name}, {node_y_name}, {node_z_name}")
            if num_faces > 0:
                print(f"  Face connectivity: {face_conn_name}")
                if face_conn_name in ds.variables:
                    face_var = ds.variables[face_conn_name]
                    print(f"    Shape: {face_var.shape}, Dtype: {face_var.dtype}")
                    print(f"    Attributes: {dict(face_var.attrs)}")
                    # Check fill value usage
                    face_data = face_var.values
                    fill_count = np.sum(face_data == fill_value)
                    print(f"    Fill values (-1): {fill_count} out of {face_data.size} total values")
                    print(f"    Min index: {np.min(face_data[face_data >= 0])}, Max index: {np.max(face_data)}")


def menu_func_import(self, context):
    self.layout.operator(ImportNetCDF.bl_idname, text="NetCDF Mesh (.nc)")


def menu_func_export(self, context):
    self.layout.operator(ExportNetCDF.bl_idname, text="NetCDF Mesh (.nc)")


def register():
    bpy.utils.register_class(ImportNetCDF)
    bpy.utils.register_class(ExportNetCDF)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(ImportNetCDF)
    bpy.utils.unregister_class(ExportNetCDF)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

def debug_print_netcdf_file(filepath):
    """Debug print the contents of a NetCDF file"""
    print(f"\n--- Debugging NetCDF File: {filepath} ---")
    verify_ds = xr.open_dataset(filepath)
    print(f"Variables in file: {sorted(verify_ds.variables.keys())}")
    print(f"Dimensions: {dict(verify_ds.dims)}")
    if topology_name in verify_ds.variables:
        verify_topo = verify_ds.variables[topology_name]
        print(f"Topology '{topology_name}' attributes: {dict(verify_topo.attrs)}")
    verify_ds.close()
    print("\n--- End of Debugging ---")

def debug_print_extension_version():
    print(f"Extension version 0")