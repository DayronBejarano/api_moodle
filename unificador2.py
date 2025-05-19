import pandas as pd
import os

# 📁 Ruta base donde están tus archivos
base_path = r"C:\Users\dayro\OneDrive\OneDrive\Documentos\dayron"

# 📥 Rutas de entrada
ruta_actividades = os.path.join(base_path, "2025-Tablero Estudiantes Actividades algoritmia.xlsx")
ruta_accesos = os.path.join(base_path, "2025-Tablero Estudiantes Accesos_algoritmia.xlsx")

# 📤 Ruta de salida
ruta_salida = os.path.join(base_path, "datos_para_modelo.csv")

# ✅ Cargar archivos
df_actividades = pd.read_excel(ruta_actividades)
df_accesos = pd.read_excel(ruta_accesos)

# 🔤 Normalizar nombres de columnas
df_actividades.columns = df_actividades.columns.str.lower().str.strip()
df_accesos.columns = df_accesos.columns.str.lower().str.strip()

# 🏷️ Renombrar 'id estudiante' si hace falta
for df in [df_actividades, df_accesos]:
    if "id estudiante" in df.columns:
        df.rename(columns={"id estudiante": "id_estudiante"}, inplace=True)

# 🔗 Unir dataframes (solo actividades y accesos)
df_merge = pd.merge(df_actividades, df_accesos, on="id_estudiante", how="outer", suffixes=('_act', '_acc'))

# 📊 Construir dataset para el modelo
df_modelo = pd.DataFrame()
df_modelo["Modo"] = df_merge.get("modo_act", 0)
df_modelo["Ingresado"] = df_merge.get("¿ha ingresado al curso alguna vez?", 0)
df_modelo["Contenido_completado"] = df_merge.get("contenidos completados [%]", 0)
df_modelo["Dias_sin_ingreso_curso"] = df_merge.get("días sin ingreso al curso desde su matrícula", 0)
df_modelo["dias_sin_ingreso_plataforma"] = df_merge.get("días sin ingreso a la plataforma", 0)
df_modelo["cuestionarios_completados"] = df_merge.get("cuestionarios completados", 0)
df_modelo["Foros_creados"] = df_merge.get("foro post creados", 0)
df_modelo["Foros_rta"] = df_merge.get("foro repuestas realizadas", 0)
df_modelo["Foros_leidos"] = df_merge.get("foro post leídos", 0)
df_modelo["Envio_buzones"] = df_merge.get("número de envíos a buzones", 0)

# 🧹 Limpiar NaNs
df_modelo = df_modelo.fillna(0)

# 💾 Guardar CSV
df_modelo.to_csv(ruta_salida, index=False, encoding='utf-8-sig')

print("✅ CSV generado correctamente en:")
print(ruta_salida)
