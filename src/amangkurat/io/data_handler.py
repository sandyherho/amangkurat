"""NetCDF data handler."""

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime


class DataHandler:
    """NetCDF output handler."""
    
    @staticmethod
    def save_netcdf(filename: str, result: dict, metadata: dict,
                   output_dir: str = "outputs"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            nx = len(result['x'])
            nt = len(result['t'])
            nc.createDimension('x', nx)
            nc.createDimension('t', nt)
            
            nc_x = nc.createVariable('x', 'f4', ('x',), zlib=True, complevel=4)
            nc_x[:] = result['x']
            nc_x.units = "dimensionless"
            nc_x.long_name = "position"
            
            nc_t = nc.createVariable('t', 'f4', ('t',), zlib=True, complevel=4)
            nc_t[:] = result['t']
            nc_t.units = "dimensionless"
            nc_t.long_name = "time"
            
            nc_phi = nc.createVariable('phi', 'f4', ('t', 'x'), zlib=True, complevel=5)
            nc_phi[:] = result['phi']
            nc_phi.long_name = "scalar_field"
            
            params = result['params']
            nc.nx = int(params['nx'])
            nc.dx = float(params['dx'])
            nc.dt_initial = float(params['dt_initial'])
            nc.dt_final = float(params['dt_final'])
            nc.potential = str(params['potential'])
            nc.n_cores = int(params['n_cores'])
            nc.adaptive_dt = int(params.get('adaptive_dt', True))
            
            nc.scenario = metadata.get('scenario_name', 'unknown')
            nc.created = datetime.now().isoformat()
            nc.software = "amangkurat-solver"
            nc.version = "0.0.1"
            nc.author = "Sandy H. S. Herho"
            nc.license = "MIT"
