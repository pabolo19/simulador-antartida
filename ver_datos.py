import pandas as pd

# Cargar el archivo limpio
archivo = 'Datos_Biologia_Completos.csv'
df = pd.read_csv(archivo)

print("\n=== RESUMEN DE LOS DATOS ===")
print(f"Total de filas: {len(df)}")
print(f"Total de columnas: {len(df.columns)}")

print("\n=== ¿QUEDAN DATOS VACÍOS (NaN)? ===")
# Esto suma los valores nulos por cada columna
print(df.isnull().sum())

print("\n=== PRIMERAS 5 FILAS (Muestra) ===")
# Forzamos a Pandas a mostrar todas las columnas sin cortarlas
pd.set_option('display.max_columns', None)
print(df.head(5))