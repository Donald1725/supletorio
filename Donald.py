from pyspark.ml.clustering import FuzzyCMeans
from pyspark.ml.feature import VectorAssembler
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Crear la sesión de Spark
spark = SparkSession.builder.appName("FuzzyCMeans").getOrCreate()

# Crear un objeto VectorAssembler para combinar las columnas relevantes en una sola columna
assembler = VectorAssembler(inputCols=['O3', 'CO', 'NO2', 'SO2', 'PM2_5'], outputCol='features')

# Definir la URL del archivo CSV
url = 'hdfs://master:9000/user/hadoop/contaminantes/datafinal.csv'

# Leer el archivo CSV y almacenarlo en un DataFrame de PySpark
df = spark.read.csv(url, header=True, inferSchema=True, sep=';', nullValue='NA')

# Transformar el DataFrame de pandas en un DataFrame de PySpark
df_spark = spark.createDataFrame(df)

# Aplicar el VectorAssembler al DataFrame de PySpark
df_assembled = assembler.transform(df_spark)

# Escalar los datos
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

num_clusters = int(input("Escriba el numero de clusteres: "))

# Crear un objeto FuzzyCMeans
fcm = FuzzyCMeans(featuresCol='scaled_features', k=num_clusters, maxIter=100, seed=42)

# Entrenar el modelo
model = fcm.fit(df_scaled)

# Obtener los centros de los clusters
centers = model.clusterCenters()

# Redondear los centros de los clusters a dos decimales
centers_rounded = [[round(x, 2) for x in center] for center in centers]

# Mostrar los resultados del modelo
print(f"El modelo se entrenó con {model.numIter} iteraciones.")
print(f"El error cuadrático medio del modelo es {round(model.computeCost(df_scaled), 2)}.")
print(f"Los centros de los clusters son: {centers_rounded}.")

# Guardar el DataFrame 'df_spark' en un archivo de csv
df_spark.write.format('com.databricks.spark.csv').option('header', 'true').save('hdfs://master:9000/user/hadoop/contaminantes/datafinal_clusters.csv')

spark.stop()