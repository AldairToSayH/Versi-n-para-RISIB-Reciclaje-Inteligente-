# SPRINT 4 - Modelo Final y Validación
import matplotlib.pyplot as plt

class FinalMLPModel:
    def __init__(self, input_shape, best_config=None):
        self.input_shape = input_shape
        self.best_config = best_config or {'lr': 0.001, 'batch_size': 32, 'units': [128, 64, 32]}
        self.final_model = None

    def crear_modelo_final(self):
        """Crear el modelo final con la mejor configuración"""
        print("🏆 CREANDO MODELO FINAL OPTIMIZADO")

        model = Sequential(name="MLP_Final_Optimizado")

        # Capa de entrada
        model.add(Dense(self.best_config['units'][0],
                       activation='relu',
                       input_shape=self.input_shape,
                       kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Capas ocultas
        for units in self.best_config['units'][1:]:
            model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))

        # Capa de salida
        model.add(Dense(1, activation='linear', kernel_initializer='glorot_normal'))

        # Compilar con mejor learning rate
        model.compile(
            optimizer=Adam(learning_rate=self.best_config['lr']),
            loss='mse',
            metrics=['mae', 'mse']
        )

        print(f"✅ Modelo final creado: {model.count_params():,} parámetros")
        self.final_model = model
        return model

    def entrenamiento_final(self, X_train, y_train, X_test, y_test, epochs=150):
        """Entrenamiento final extensivo del modelo"""
        print("\n🎯 ENTRENAMIENTO FINAL DEL MODELO")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            ModelCheckpoint('/content/mlp_final_model.h5', monitor='val_loss',
                          save_best_only=True)
        ]

        history = self.final_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=self.best_config['batch_size'],
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        return history

    def evaluacion_completa(self, X_test, y_test):
        """Evaluación exhaustiva del modelo final"""
        print("\n📊 EVALUACIÓN COMPLETA DEL MODELO FINAL")

        # Predicciones
        y_pred = self.final_model.predict(X_test, verbose=0).flatten()

        # Métricas principales
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print(f"🔍 MÉTRICAS PRINCIPALES:")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")

        # Análisis de errores
        errors = y_test - y_pred
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'within_5%': np.mean(np.abs(errors) < 0.05 * y_test) * 100,
            'within_10%': np.mean(np.abs(errors) < 0.10 * y_test) * 100
        }

        print(f"\n📈 ANÁLISIS DE ERRORES:")
        print(f"   Error promedio: {error_stats['mean_error']:.4f}")
        print(f"   Desviación estándar error: {error_stats['std_error']:.4f}")
        print(f"   Error máximo: {error_stats['max_error']:.4f}")
        print(f"   Dentro del 5%: {error_stats['within_5%']:.2f}%")
        print(f"   Dentro del 10%: {error_stats['within_10%']:.2f}%")

        return {
            'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape},
            'error_analysis': error_stats,
            'predictions': y_pred,
            'errors': errors
        }

    def visualizar_resultados(self, history, evaluation_results, y_test, y_pred):
        """Visualización completa de resultados"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Resultados Finales - Modelo MLP Optimizado', fontsize=16)

        # 1. Pérdida durante entrenamiento
        axes[0,0].plot(history.history['loss'], label='Train Loss', alpha=0.7)
        axes[0,0].plot(history.history['val_loss'], label='Val Loss', alpha=0.7)
        axes[0,0].set_title('Pérdida durante Entrenamiento')
        axes[0,0].set_xlabel('Época')
        axes[0,0].set_ylabel('Pérdida (MSE)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. MAE durante entrenamiento
        axes[0,1].plot(history.history['mae'], label='Train MAE', alpha=0.7)
        axes[0,1].plot(history.history['val_mae'], label='Val MAE', alpha=0.7)
        axes[0,1].set_title('MAE durante Entrenamiento')
        axes[0,1].set_xlabel('Época')
        axes[0,1].set_ylabel('MAE')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. Predicciones vs Valores Reales
        axes[0,2].scatter(y_test, y_pred, alpha=0.6, s=20)
        axes[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,2].set_xlabel('Valores Reales')
        axes[0,2].set_ylabel('Predicciones')
        axes[0,2].set_title(f'Predicciones vs Reales\nR² = {evaluation_results["metrics"]["R2"]:.4f}')
        axes[0,2].grid(True, alpha=0.3)

        # 4. Distribución de errores
        errors = y_test - y_pred
        axes[1,0].hist(errors, bins=30, alpha=0.7, color='orange')
        axes[1,0].axvline(0, color='red', linestyle='--')
        axes[1,0].set_xlabel('Error')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].set_title('Distribución de Errores')
        axes[1,0].grid(True, alpha=0.3)

        # 5. Residual plot
        axes[1,1].scatter(y_pred, errors, alpha=0.6, s=20)
        axes[1,1].axhline(0, color='red', linestyle='--')
        axes[1,1].set_xlabel('Predicciones')
        axes[1,1].set_ylabel('Residuales')
        axes[1,1].set_title('Gráfico de Residuales')
        axes[1,1].grid(True, alpha=0.3)

        # 6. Métricas comparativas
        metrics_names = ['MAE', 'RMSE', 'R2']
        metrics_values = [
            evaluation_results['metrics']['MAE'],
            evaluation_results['metrics']['RMSE'],
            evaluation_results['metrics']['R2']
        ]

        bars = axes[1,2].bar(metrics_names, metrics_values,
                            color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1,2].set_title('Métricas del Modelo')
        axes[1,2].set_ylabel('Valor')

        # Añadir valores en las barras
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            axes[1,2].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

# Ejecutar Sprint 4
print("\n" + "="*50)
print("SPRINT 4: MODELO FINAL Y VALIDACIÓN")
print("="*50)

# Crear y entrenar modelo final
final_model_builder = FinalMLPModel(input_shape=(X_train.shape[1],), best_config=best_config)
final_model = final_model_builder.crear_modelo_final()

# Entrenamiento final
final_history = final_model_builder.entrenamiento_final(X_train, y_train, X_test, y_test)

# Evaluación completa
final_evaluation = final_model_builder.evaluacion_completa(X_test, y_test)

# Comparación final con todos los modelos
print("\n" + "="*60)
print("COMPARACIÓN FINAL DE TODOS LOS MODELOS")
print("="*60)

# Añadir modelo final a la comparación
y_pred_final = final_evaluation['predictions']
final_metrics = final_evaluation['metrics']

trainer.results['MLP_Final'] = {
    'MAE': final_metrics['MAE'],
    'RMSE': final_metrics['RMSE'],
    'R2': final_metrics['R2'],
    'Error_Porcentual': final_metrics['MAPE'],
    'y_pred': y_pred_final
}

comparison_final = trainer.comparar_modelos()

# Visualizar resultados finales
final_model_builder.visualizar_resultados(
    final_history, final_evaluation, y_test, y_pred_final
)

print("\n🎉 FASE 4 - MODELADO COMPLETADA EXITOSAMENTE!")
