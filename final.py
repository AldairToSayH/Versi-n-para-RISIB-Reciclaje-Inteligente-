# SOLUCION_DEFINITIVA_CORREGIDA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("🚀 SOLUCIÓN DEFINITIVA - CORREGIDA")

# =============================================================================
# PASO 1: CREAR DATOS COHERENTES Y REENTRENAR MODELO
# =============================================================================

def crear_dataset_coherente():
    """Crear un dataset sintético pero coherente para el modelo"""
    print("📊 CREANDO DATASET COHERENTE...")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Características REALES que usaremos (7 características significativas)
    caracteristicas_reales = [
        'distancia_km',
        'hora_dia', 
        'dia_semana',
        'nivel_lluvia',
        'nivel_trafico',
        'temperatura',
        'es_fin_semana'
    ]
    
    # Generar datos sintéticos realistas
    data = {
        'distancia_km': np.random.uniform(2, 50, n_samples),
        'hora_dia': np.random.randint(0, 24, n_samples),
        'dia_semana': np.random.randint(0, 7, n_samples),
        'nivel_lluvia': np.random.uniform(0, 1, n_samples),
        'nivel_trafico': np.random.uniform(0, 1, n_samples),
        'temperatura': np.random.uniform(5, 35, n_samples),
        'es_fin_semana': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Crear variable objetivo realista (tiempo de viaje en minutos)
    # Tiempo base + efectos de tráfico + efectos de lluvia + efectos horarios
    tiempo_base = df['distancia_km'] * 2  # 2 min/km en condiciones ideales
    
    # Efecto del tráfico (puede duplicar el tiempo)
    efecto_trafico = df['nivel_trafico'] * tiempo_base * 0.8
    
    # Efecto de la lluvia
    efecto_lluvia = df['nivel_lluvia'] * tiempo_base * 0.3
    
    # Efecto de hora pico
    hora_pico = ((df['hora_dia'] >= 7) & (df['hora_dia'] <= 9)) | ((df['hora_dia'] >= 16) & (df['hora_dia'] <= 19))
    efecto_hora_pico = hora_pico.astype(int) * tiempo_base * 0.4
    
    # Efecto fin de semana (menos tráfico generalmente)
    efecto_fin_semana = df['es_fin_semana'] * tiempo_base * (-0.2)
    
    # Tiempo total
    df['tiempo_viaje_min'] = tiempo_base + efecto_trafico + efecto_lluvia + efecto_hora_pico + efecto_fin_semana
    
    # Agregar algo de ruido
    df['tiempo_viaje_min'] += np.random.normal(0, 5, n_samples)
    
    # Asegurar valores positivos
    df['tiempo_viaje_min'] = df['tiempo_viaje_min'].clip(lower=5)
    
    print(f"✅ Dataset creado: {df.shape}")
    print(f"   Características: {caracteristicas_reales}")
    print(f"   Tiempo de viaje - Min: {df['tiempo_viaje_min'].min():.1f}, Max: {df['tiempo_viaje_min'].max():.1f}")
    
    return df, caracteristicas_reales

# Crear dataset coherente
df, caracteristicas_reales = crear_dataset_coherente()

# =============================================================================
# PASO 2: PREPARAR DATOS Y ENTRENAR MODELO
# =============================================================================

def entrenar_modelo_coherente(df, caracteristicas):
    """Entrenar modelo MLP con datos coherentes"""
    print("\n🎯 ENTRENANDO MODELO COHERENTE...")
    
    # Preparar características y objetivo
    X = df[caracteristicas].values
    y = df['tiempo_viaje_min'].values
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Datos preparados:")
    print(f"   • X_train: {X_train_scaled.shape}")
    print(f"   • X_test: {X_test_scaled.shape}")
    print(f"   • Características: {caracteristicas}")
    
    # Crear y entrenar modelo MLP simple pero efectivo
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Entrenamiento rápido
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluar modelo
    train_score = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_score = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"✅ Modelo entrenado:")
    print(f"   • Train MAE: {train_score[1]:.2f} minutos")
    print(f"   • Test MAE: {test_score[1]:.2f} minutos")
    
    return model, scaler, X_test_scaled, y_test, history

# Entrenar modelo coherente
modelo_coherente, scaler_coherente, X_test_coherente, y_test_coherente, history_coherente = entrenar_modelo_coherente(df, caracteristicas_reales)

# =============================================================================
# PASO 3: SISTEMA DE PREDICCIÓN COHERENTE (CORREGIDO)
# =============================================================================

class SistemaPrediccionCoherente:
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.historial_predicciones = []
        print(f"✅ Sistema coherente inicializado con {len(feature_names)} características")
        
    def preparar_datos_entrada(self, datos_usuario):
        """Preparar datos de entrada de forma coherente"""
        # Crear array con todas las características en el orden correcto
        caracteristicas_array = np.zeros(len(self.feature_names))
        
        # Mapeo de datos de usuario a características del modelo
        mapeo = {
            'distancia': 'distancia_km',
            'hora': 'hora_dia',
            'dia_semana': 'dia_semana', 
            'lluvia': 'nivel_lluvia',
            'trafico_intenso': 'nivel_trafico',
            'temperatura': 'temperatura'
        }
        
        # Asignar valores
        for clave_usuario, clave_modelo in mapeo.items():
            if clave_usuario in datos_usuario and clave_modelo in self.feature_names:
                idx = self.feature_names.index(clave_modelo)
                caracteristicas_array[idx] = datos_usuario[clave_usuario]
        
        # Calcular es_fin_semana automáticamente
        if 'es_fin_semana' in self.feature_names:
            idx = self.feature_names.index('es_fin_semana')
            dia_semana = datos_usuario.get('dia_semana', 0)
            caracteristicas_array[idx] = 1 if dia_semana >= 5 else 0
        
        return caracteristicas_array
    
    def predecir_tiempo_viaje(self, datos_usuario):
        """Predecir tiempo de viaje"""
        try:
            print(f"\n🎯 REALIZANDO PREDICCIÓN...")
            print(f"   Datos entrada: {datos_usuario}")
            
            # Preparar características
            caracteristicas_array = self.preparar_datos_entrada(datos_usuario)
            print(f"   Características preparadas: {caracteristicas_array}")
            
            # Verificar dimensiones
            if len(caracteristicas_array) != len(self.feature_names):
                raise ValueError(f"Dimensiones incorrectas: {len(caracteristicas_array)} != {len(self.feature_names)}")
            
            # Escalar
            caracteristicas_esc = self.scaler.transform([caracteristicas_array])
            print(f"   Características escaladas: {caracteristicas_esc.shape}")
            
            # Predecir
            tiempo_predicho = self.model.predict(caracteristicas_esc, verbose=0)[0][0]
            tiempo_predicho = max(5.0, tiempo_predicho)  # Mínimo 5 minutos
            
            print(f"✅ Predicción exitosa: {tiempo_predicho:.1f} minutos")
            
            # Guardar en historial
            self.historial_predicciones.append({
                'timestamp': datetime.now(),
                'datos_entrada': datos_usuario.copy(),
                'tiempo_predicho': tiempo_predicho
            })
            
            return tiempo_predicho
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None
    
    def generar_recomendaciones(self, datos_usuario, tiempo_predicho):
        """Generar recomendaciones inteligentes"""
        recomendaciones = []
        distancia = datos_usuario.get('distancia', 10)
        hora = datos_usuario.get('hora', 12)
        lluvia = datos_usuario.get('lluvia', 0)
        trafico = datos_usuario.get('trafico_intenso', 0)
        
        # Calcular velocidad promedio
        velocidad = (distancia / tiempo_predicho) * 60
        
        # Análisis de condiciones
        if velocidad < 15:
            recomendaciones.append("🚗 Tráfico MUY intenso - Considere posponer el viaje")
            recomendaciones.append("⏰ Salga con 20-30 minutos de anticipación")
        elif velocidad < 25:
            recomendaciones.append("🚙 Tráfico intenso - Espere demoras")
            recomendaciones.append("📱 Revise rutas alternativas en su app de navegación")
        elif velocidad < 40:
            recomendaciones.append("✅ Tráfico normal - Tiempo de viaje estándar")
        else:
            recomendaciones.append("🎯 Condiciones óptimas - Buen tiempo para viajar")
        
        if lluvia > 0.7:
            recomendaciones.append("🌧️ Lluvia intensa - Extreme precauciones")
        elif lluvia > 0.3:
            recomendaciones.append("🌦️ Lluvia moderada - Conduzca con cuidado")
        
        if trafico > 0.8:
            recomendaciones.append("🚦 Congestión severa - Evite horas pico si es posible")
        
        if 7 <= hora <= 9:
            recomendaciones.append("🏙️ Hora pico matutina - Zonas escolares y comerciales congestionadas")
        elif 16 <= hora <= 19:
            recomendaciones.append("🌆 Hora pico vespertina - Tráfico pesado en vías principales")
        
        return recomendaciones, velocidad
    
    def mostrar_prediccion(self, datos_usuario, tiempo_predicho, recomendaciones, velocidad):
        """Mostrar resultados de forma clara"""
        print("\n" + "="*60)
        print("🎯 RESULTADO DE PREDICCIÓN")
        print("="*60)
        
        print(f"\n📋 INFORMACIÓN DEL VIAJE:")
        print(f"   • Distancia: {datos_usuario.get('distancia', 'N/A')} km")
        print(f"   • Hora de salida: {datos_usuario.get('hora', 'N/A')}:00")
        print(f"   • Día: {self.obtener_dia(datos_usuario.get('dia_semana', 0))}")
        print(f"   • Condiciones: {'Lluvia' if datos_usuario.get('lluvia', 0) > 0.3 else 'Seco'} | "
              f"{'Tráfico intenso' if datos_usuario.get('trafico_intenso', 0) > 0.5 else 'Tráfico fluido'}")
        
        print(f"\n⏰ PREDICCIÓN:")
        print(f"   • Tiempo estimado: {tiempo_predicho:.1f} minutos")
        print(f"   • Velocidad promedio: {velocidad:.1f} km/h")
        
        print(f"\n💡 RECOMENDACIONES:")
        for i, rec in enumerate(recomendaciones, 1):
            print(f"   {i}. {rec}")
        
        print(f"\n🕒 Hora estimada de llegada: {self.calcular_hora_llegada(tiempo_predicho)}")
    
    def obtener_dia(self, dia_num):
        dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        return dias[dia_num] if 0 <= dia_num < 7 else 'Desconocido'
    
    def calcular_hora_llegada(self, minutos):
        """Calcular hora estimada de llegada - CORREGIDO"""
        # Convertir a float nativo de Python para evitar error con numpy.float32
        minutos_float = float(minutos)
        llegada = datetime.now() + timedelta(minutes=minutos_float)
        return llegada.strftime("%H:%M")
    
    def interfaz_usuario(self):
        """Interfaz amigable para el usuario"""
        print("\n" + "="*60)
        print("🚗 PREDICCIÓN INTELIGENTE DE TIEMPOS DE VIAJE")
        print("="*60)
        print("\nIngrese los datos de su viaje:")
        
        try:
            datos = {
                'distancia': float(input("\n📏 Distancia del viaje (km): ")),
                'hora': int(input("🕒 Hora de salida (0-23): ")),
                'dia_semana': int(input("📅 Día de semana (0=Lunes, 6=Domingo): ")),
                'lluvia': float(input("🌧️ Nivel de lluvia (0-1, donde 1=llueve mucho): ")),
                'trafico_intenso': float(input("🚦 Nivel de tráfico (0-1, donde 1=muy congestionado): ")),
                'temperatura': float(input("🌡️ Temperatura (°C): "))
            }
            
            # Realizar predicción
            tiempo = self.predecir_tiempo_viaje(datos)
            
            if tiempo is not None:
                # Generar recomendaciones
                recomendaciones, velocidad = self.generar_recomendaciones(datos, tiempo)
                
                # Mostrar resultados
                self.mostrar_prediccion(datos, tiempo, recomendaciones, velocidad)
                return tiempo, recomendaciones
            else:
                print("❌ No se pudo realizar la predicción")
                return None, None
                
        except ValueError:
            print("❌ Error: Ingrese valores numéricos válidos")
            return None, None
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            return None, None

# =============================================================================
# PASO 4: INICIALIZAR SISTEMA COHERENTE
# =============================================================================

print("\n🚀 INICIALIZANDO SISTEMA COHERENTE...")
sistema_coherente = SistemaPrediccionCoherente(
    model=modelo_coherente,
    scaler=scaler_coherente, 
    feature_names=caracteristicas_reales
)

print("✅ Sistema coherente listo para usar!")

# =============================================================================
# PASO 5: PRUEBAS AUTOMÁTICAS COHERENTES
# =============================================================================

def ejecutar_pruebas_coherentes():
    """Ejecutar pruebas con datos coherentes"""
    print("\n" + "="*70)
    print("🧪 EJECUTANDO PRUEBAS COHERENTES")
    print("="*70)
    
    casos_prueba = [
        {
            'nombre': "🚗 Lunes 8AM - Tráfico Intenso",
            'datos': {'distancia': 15, 'hora': 8, 'dia_semana': 0, 'lluvia': 0.2, 'trafico_intenso': 0.9, 'temperatura': 18},
            'desc': "Viaje matutino en día laboral con mucho tráfico"
        },
        {
            'nombre': "🚙 Miércoles 2PM - Tráfico Fluido", 
            'datos': {'distancia': 20, 'hora': 14, 'dia_semana': 2, 'lluvia': 0.0, 'trafico_intenso': 0.1, 'temperatura': 25},
            'desc': "Viaje vespertino en condiciones ideales"
        },
        {
            'nombre': "🌧️ Sábado Noche - Lluvia Intensa",
            'datos': {'distancia': 8, 'hora': 21, 'dia_semana': 5, 'lluvia': 0.9, 'trafico_intenso': 0.3, 'temperatura': 12},
            'desc': "Viaje corto nocturno con lluvia intensa"
        },
        {
            'nombre': "🏙️ Viernes 6PM - Hora Pico",
            'datos': {'distancia': 12, 'hora': 18, 'dia_semana': 4, 'lluvia': 0.1, 'trafico_intenso': 0.95, 'temperatura': 22},
            'desc': "Viernes en hora pico vespertina"
        }
    ]
    
    resultados = []
    
    for caso in casos_prueba:
        print(f"\n🔍 Probando: {caso['nombre']}")
        print(f"   {caso['desc']}")
        
        tiempo = sistema_coherente.predecir_tiempo_viaje(caso['datos'])
        
        if tiempo is not None:
            recom, vel = sistema_coherente.generar_recomendaciones(caso['datos'], tiempo)
            resultados.append({
                'caso': caso['nombre'],
                'tiempo': tiempo,
                'velocidad': vel,
                'recomendacion': recom[0] if recom else "Sin recomendación"
            })
            
            print(f"   ✅ Tiempo: {tiempo:.1f} min | Velocidad: {vel:.1f} km/h")
        else:
            print("   ❌ Error en predicción")
    
    # Mostrar resumen
    print("\n" + "="*70)
    print("📊 RESUMEN DE PRUEBAS")
    print("="*70)
    
    for res in resultados:
        print(f"\n📋 {res['caso']}")
        print(f"   • Tiempo: {res['tiempo']:.1f} min")
        print(f"   • Velocidad: {res['velocidad']:.1f} km/h") 
        print(f"   • Recomendación: {res['recomendacion']}")
    
    return resultados

# Ejecutar pruebas coherentes
resultados_pruebas = ejecutar_pruebas_coherentes()

# =============================================================================
# PASO 6: INTERFAZ FINAL COHERENTE
# =============================================================================

def menu_principal_coherente():
    """Menú principal del sistema coherente"""
    while True:
        print("\n" + "="*60)
        print("🚗 SISTEMA INTELIGENTE DE PREDICCIÓN DE RUTAS")
        print("="*60)
        print("\nOpciones:")
        print("1. 🎯 Realizar predicción de viaje")
        print("2. 🧪 Ver ejemplos de prueba")
        print("3. 📊 Ver información del sistema")
        print("4. 🚪 Salir")
        
        opcion = input("\nSeleccione opción (1-4): ")
        
        if opcion == '1':
            sistema_coherente.interfaz_usuario()
        elif opcion == '2':
            ejecutar_pruebas_coherentes()
        elif opcion == '3':
            print(f"\n📊 INFORMACIÓN DEL SISTEMA:")
            print(f"   • Modelo: MLP con {len(caracteristicas_reales)} características")
            print(f"   • Características: {caracteristicas_reales}")
            print(f"   • Predicciones realizadas: {len(sistema_coherente.historial_predicciones)}")
        elif opcion == '4':
            print("👋 ¡Gracias por usar nuestro sistema!")
            break
        else:
            print("❌ Opción no válida")

# =============================================================================
# EJECUCIÓN FINAL
# =============================================================================

print("\n🎉 ¡SISTEMA COHERENTE COMPLETAMENTE FUNCIONAL!")
print("   • Dataset coherente ✓")
print("   • Modelo reentrenado ✓") 
print("   • Scaler compatible ✓")
print("   • Sistema de predicción ✓")
print("   • Pruebas automáticas ✓")
print("   • Error timedelta corregido ✓")

# Iniciar sistema
menu_principal_coherente()
