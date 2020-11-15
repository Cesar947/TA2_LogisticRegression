package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"gonum.org/v1/gonum/mat"
)

//Estructura que convierte un dataframe en una matriz
type matrix struct {
	dataframe.DataFrame
}

func (m matrix) At(i, j int) float64 {
	return m.Elem(i, j).Float()
}

func (m matrix) T() mat.Matrix {
	return mat.Transpose{m}
}

type RegresionLogistica struct {
	factorAprendizaje float64
	pesos             mat.Matrix
	iteraciones       int
	umbral            float64
}

func inicializarPesos(n_elementos_entrada int) mat.Matrix {
	arreglo_pesos := make([]float64, n_elementos_entrada)
	rand.Seed(42)
	for i := 0; i < n_elementos_entrada; i++ {
		arreglo_pesos[i] = rand.Float64()
	}

	pesos := mat.NewDense(n_elementos_entrada, 1, arreglo_pesos)
	return pesos
}

func sigmoide(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func factorUmbral(matriz_resta mat.Matrix) float64 {
	var sum float64
	valores := mat.Col(nil, 0, matriz_resta)
	for _, value := range valores {
		sum += value
	}
	return sum
}

//Para la precisión. Cuenta los números de la matriz que no son ceros para saber cuántos resultados fueorn incorrectos
func contadorNoCeros(matriz_resta mat.Matrix) int {
	var count int
	valores := mat.Col(nil, 0, matriz_resta)
	for _, value := range valores {
		if value != 0 {
			count += 1
		}
	}
	return count
}

func crearPredicciones(x mat.Matrix, umbral float64) mat.Matrix {
	outputs := mat.Col(nil, 0, x)
	size := len(outputs)
	var modelo_lineal float64
	process_outputs := make([]float64, size)
	for i, value := range outputs {
		modelo_lineal = value + umbral
		process_outputs[i] = sigmoide(modelo_lineal)
	}

	return mat.NewDense(size, 1, process_outputs)
}

//Función que hace los calculos del modelo lineal y la función sigmoide
func (regresion *RegresionLogistica) predecir(entradas mat.Matrix) mat.Matrix {

	//Resultados modelo lineal
	filas, _ := entradas.Dims()
	_, columnas := regresion.pesos.Dims()
	w := mat.NewDense(filas, columnas, nil)

	w.Product(entradas, regresion.pesos)

	predicciones := crearPredicciones(w, regresion.umbral)

	return predicciones
}

//Función de costo para poder definir una configuración de coeficientes óptima
func funcion_costo(etiquetas, predicciones mat.Matrix) float64 {

	vector_etiquetas := mat.Col(nil, 0, etiquetas)
	vector_predicciones := mat.Col(nil, 0, predicciones)

	//Misma longitud entre etiquetas y predicciones
	length := len(vector_etiquetas)
	//Error cuando la etiqueta es 1
	costo1 := 0.0
	//Error cuando la etiqueta es 0
	costo0 := 0.0
	for i := 0; i < length; i++ {
		if vector_etiquetas[i] == 1 {
			costo1 += -1 * (math.Log(vector_predicciones[i]))
		} else if vector_etiquetas[i] == 0 {
			costo0 += -1 * (math.Log(1 - vector_predicciones[i]))
		}

	}
	costo := (costo1 + costo0) / float64(length)

	return costo

}

//Actualiza los coeficientes o pesos del algoritmo a través del método de la gradiente descendiente
func (regresion *RegresionLogistica) actualizarPesos(entradas, etiquetas, predicciones mat.Matrix) {
	//Se multiplicará la traspuesto de la matriz de entradas por
	// la resta entre las matrices predicciones y etiquetas.
	//entradas.Dims() = (891 7)
	//predicciones.Dims() = etiquetas.Dims() = (891 1)
	//entradas.T().Dims() = (7 891)
	//matriz_gradiente.Dims() = pesos.Dims() = (7 1)

	lr := regresion.factorAprendizaje

	n_entradas, filas_pesos := entradas.Dims()
	resta_predicciones_etiquetas := mat.NewDense(n_entradas, 1, nil)

	resta_predicciones_etiquetas.Sub(predicciones, etiquetas)

	gradiente := mat.NewDense(filas_pesos, 1, nil)
	gradiente.Product(entradas.T(), resta_predicciones_etiquetas)

	valores_gradiente := mat.Col(nil, 0, gradiente)

	valores_pesos := mat.Col(nil, 0, regresion.pesos)
	for i, value := range valores_gradiente {

		valores_pesos[i] -= lr * (value / float64(n_entradas))

	}

	regresion.pesos = mat.NewDense(filas_pesos, 1, valores_pesos)
	regresion.umbral -= lr * (factorUmbral(resta_predicciones_etiquetas) / float64(n_entradas))

}

//Entrenar al algoritmo. Repetir, de acuerdo al número de iteraciones, las funciones "predecir" y "actualizar pesos"
func (regresion *RegresionLogistica) entrenar(entradas, etiquetas mat.Matrix) mat.Matrix {
	var predicciones mat.Matrix

	for i := 0; i < regresion.iteraciones; i++ {

		predicciones = regresion.predecir(entradas)
		regresion.actualizarPesos(entradas, etiquetas, predicciones)

	}

	return predicciones

}

//Convertir valores continuos del sigmoide en 1 y 0 para la clasificación
func clasificar(predicciones mat.Matrix) mat.Matrix {
	valores := mat.Col(nil, 0, predicciones)

	for i, valor := range valores {
		if valor < 0.5 {
			valores[i] = 0
		} else {
			valores[i] = 1
		}
	}
	matriz_clasificacion := mat.NewDense(len(valores), 1, valores)
	return matriz_clasificacion
}

//Calcular los resultados del dataset de prueba
func (regresion *RegresionLogistica) testear(entradas_test mat.Matrix) mat.Matrix {

	predicciones_test := regresion.predecir(entradas_test)
	return clasificar(predicciones_test)
}

func precision(predicciones, etiquetas mat.Matrix) float64 {
	filas, _ := etiquetas.Dims()
	resta_predicciones_etiquetas := mat.NewDense(filas, 1, nil)
	resta_predicciones_etiquetas.Sub(predicciones, etiquetas)
	return 1.0 - float64(contadorNoCeros(resta_predicciones_etiquetas))/float64(filas)

}

func imprimirMatriz(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func errorHandler(err error) {
	if err != nil {
		fmt.Println("Error encontrado ::", err)
		return
	}
}

func main() {

	archivo, err := os.Open("./titanic_data_modified.csv")
	errorHandler(err)

	df := dataframe.ReadCSV(archivo)
	matriz_inicial := matrix{df}
	cant_filas, _ := matriz_inicial.Dims()
	cantidad_corte := float64(cant_filas) * 0.8

	//Separación del dataset en entrenamiento y pruebas
	var matriz_entradas_train, matriz_entradas_test, matriz_etiquetas_train, matriz_etiquetas_test mat.Matrix

	train := df.Filter(dataframe.F{"PassengerId", series.LessEq, cantidad_corte})
	test := df.Filter(dataframe.F{"PassengerId", series.Greater, cantidad_corte})

	var waitGroup sync.WaitGroup
	waitGroup.Add(1)
	go func() {
		entradas_train := train.Select([]string{"Pclass", "Sex", "SibSp", "Fare", "has_C", "has_Q", "has_S"})
		entradas_test := test.Select([]string{"Pclass", "Sex", "SibSp", "Fare", "has_C", "has_Q", "has_S"})
		etiquetas_train := train.Select([]string{"Survived"})
		etiquetas_test := test.Select([]string{"Survived"})

		matriz_entradas_train = matrix{entradas_train}
		matriz_entradas_test = matrix{entradas_test}
		matriz_etiquetas_train = matrix{etiquetas_train}
		matriz_etiquetas_test = matrix{etiquetas_test}
		waitGroup.Done()
	}()
	waitGroup.Wait()

	//Inicializando los pesos
	_, n_entradas := matriz_entradas_train.Dims()
	pesos_iniciales := inicializarPesos(n_entradas)

	//Inicializando la estructura de regresión logística
	reg := RegresionLogistica{0.5, pesos_iniciales, 6000, 1}
	//resultados_entrenamiento :=

	//Entranamiento
	reg.entrenar(matriz_entradas_train, matriz_etiquetas_train)

	resultados_prueba := reg.testear(matriz_entradas_test)
	fmt.Println("Predicciones")
	imprimirMatriz(resultados_prueba)
	precision_test := precision(resultados_prueba, matriz_etiquetas_test)
	fmt.Println("Accuracy Test:", precision_test*100, "%")
}
