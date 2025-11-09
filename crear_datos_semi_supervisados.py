#!/usr/bin/env python3
"""
Script para crear datos semi-supervisados artificiales 
removiendo etiquetas de algunos registros aleatoriamente
"""

import asyncio
import sys
import random
from pathlib import Path

# Agregar el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent))

async def create_semi_supervised_data():
    """Crea datos semi-supervisados removiendo algunas etiquetas"""
    
    try:
        from app.config.connection import get_connection
        
        print("🔄 Conectando a PostgreSQL...")
        connection = await get_connection()
        
        # Obtener conteo actual
        result = await connection.fetch("SELECT COUNT(*) as total FROM postulaciones WHERE estado IS NOT NULL")
        total_labeled = result[0]['total']
        
        print(f"📊 Total de postulaciones etiquetadas: {total_labeled}")
        
        if total_labeled == 0:
            print("❌ No hay datos etiquetados para modificar")
            return
        
        # Determinar cuántas etiquetas remover (20-30% para simular semi-supervisado)
        removal_percentage = 0.25  # 25%
        records_to_unlabel = int(total_labeled * removal_percentage)
        
        print(f"🎯 Removiendo etiquetas de {records_to_unlabel} registros ({removal_percentage*100:.1f}%)")
        
        # Seleccionar IDs aleatorios para remover etiquetas
        ids_result = await connection.fetch("SELECT id FROM postulaciones WHERE estado IS NOT NULL ORDER BY RANDOM() LIMIT $1", records_to_unlabel)
        ids_to_unlabel = [row['id'] for row in ids_result]
        
        # Respaldar los estados originales (opcional, para poder restaurar después)
        backup_file = "labeled_states_backup.txt"
        print(f"💾 Creando respaldo de estados en {backup_file}")
        
        backup_data = []
        for id_val in ids_to_unlabel:
            state_result = await connection.fetch("SELECT id, estado FROM postulaciones WHERE id = $1", id_val)
            if state_result:
                backup_data.append(f"{id_val},{state_result[0]['estado']}")
        
        with open(backup_file, 'w') as f:
            f.write("id,estado_original\\n")
            f.write("\\n".join(backup_data))
        
        # Remover etiquetas (establecer estado como NULL)
        print("🔄 Removiendo etiquetas...")
        for id_val in ids_to_unlabel:
            await connection.execute("UPDATE postulaciones SET estado = NULL WHERE id = $1", id_val)
        
        # Verificar resultado
        labeled_result = await connection.fetch("SELECT COUNT(*) as labeled FROM postulaciones WHERE estado IS NOT NULL")
        unlabeled_result = await connection.fetch("SELECT COUNT(*) as unlabeled FROM postulaciones WHERE estado IS NULL")
        
        labeled_count = labeled_result[0]['labeled']
        unlabeled_count = unlabeled_result[0]['unlabeled']
        
        print("\\n✅ Datos semi-supervisados creados exitosamente:")
        print(f"📊 Postulaciones etiquetadas: {labeled_count}")
        print(f"📊 Postulaciones no etiquetadas: {unlabeled_count}")
        print(f"📊 Porcentaje etiquetado: {(labeled_count/(labeled_count+unlabeled_count)*100):.1f}%")
        print(f"💾 Respaldo guardado en: {backup_file}")
        
        await connection.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def restore_labels():
    """Restaura las etiquetas desde el archivo de respaldo"""
    
    backup_file = "labeled_states_backup.txt"
    
    if not Path(backup_file).exists():
        print(f"❌ Archivo de respaldo {backup_file} no encontrado")
        return
    
    try:
        from app.config.connection import get_connection
        
        print("🔄 Conectando a PostgreSQL...")
        connection = await get_connection()
        
        print(f"📄 Leyendo respaldo desde {backup_file}")
        
        restored_count = 0
        with open(backup_file, 'r') as f:
            lines = f.readlines()[1:]  # Saltar encabezado
            
            for line in lines:
                if line.strip():
                    id_val, estado_original = line.strip().split(',', 1)
                    await connection.execute(
                        "UPDATE postulaciones SET estado = $1 WHERE id = $2", 
                        estado_original, id_val
                    )
                    restored_count += 1
        
        print(f"✅ Restauradas {restored_count} etiquetas")
        
        # Verificar
        labeled_result = await connection.fetch("SELECT COUNT(*) as labeled FROM postulaciones WHERE estado IS NOT NULL")
        unlabeled_result = await connection.fetch("SELECT COUNT(*) as unlabeled FROM postulaciones WHERE estado IS NULL")
        
        labeled_count = labeled_result[0]['labeled']
        unlabeled_count = unlabeled_result[0]['unlabeled']
        
        print(f"📊 Estado actual:")
        print(f"  - Etiquetadas: {labeled_count}")
        print(f"  - No etiquetadas: {unlabeled_count}")
        
        await connection.close()
        
    except Exception as e:
        print(f"❌ Error restaurando etiquetas: {e}")

def print_help():
    """Muestra la ayuda del script"""
    print("""
🤖 Script para Datos Semi-Supervisados

Uso:
  python crear_datos_semi_supervisados.py create    # Crear datos semi-supervisados
  python crear_datos_semi_supervisados.py restore   # Restaurar etiquetas originales
  python crear_datos_semi_supervisados.py help      # Mostrar esta ayuda

Descripción:
  - 'create': Remueve aleatoriamente ~25% de las etiquetas para simular aprendizaje semi-supervisado
  - 'restore': Restaura las etiquetas originales desde el archivo de respaldo
  - Un archivo de respaldo se crea automáticamente antes de remover etiquetas

Archivos generados:
  - labeled_states_backup.txt: Respaldo de estados originales
""")

async def main():
    """Función principal"""
    
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'create':
        print("🚀 CREANDO DATOS SEMI-SUPERVISADOS")
        print("=" * 50)
        await create_semi_supervised_data()
        
        print("\\n" + "=" * 50)
        print("✨ ¡Listo! Ahora puedes entrenar el modelo semi-supervisado:")
        print("   python train_semi_supervised_step_by_step.py")
        
    elif command == 'restore':
        print("🔄 RESTAURANDO ETIQUETAS ORIGINALES")
        print("=" * 50)
        await restore_labels()
        
    elif command == 'help':
        print_help()
        
    else:
        print(f"❌ Comando no reconocido: {command}")
        print_help()

if __name__ == "__main__":
    asyncio.run(main())