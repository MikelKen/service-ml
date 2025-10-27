#!/usr/bin/env python3
"""
Script para ejecutar el servicio ML (FastAPI)
"""
import subprocess
import sys
import os

def main():
    """Ejecutar el servicio ML"""
    try:
        # Cambiar al directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Ejecutar la aplicación FastAPI
        subprocess.run([sys.executable, "-m", "app.main"], check=True)
    except KeyboardInterrupt:
        print("\nServicio ML detenido por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el servicio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()