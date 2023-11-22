library(readxl)
library(caret)
library(class)
library(dplyr)
library(gmodels)
library(e1071)
library(rpart)

# Lista as worksheet no arquivo Excel
excel_sheets("Acoustic_Extinguisher_Fire_Dataset.xlsx")

# Importando uma worksheet para um dataframe
dados <- read_excel("Acoustic_Extinguisher_Fire_Dataset.xlsx", sheet = 1)

# Verificando ocorrência de valores NA
colSums(is.na(dados))

# Dimensões
dim(dados)

# Tipos de dados
str(dados)

# Convertentendo variaveis para o tipo fator
dados$STATUS <- as.factor(dados$STATUS)
dados$FUEL <- as.factor(dados$FUEL)
str(dados)

# Medidas de Tendência Central
summary(dados[c("DISTANCE", "DESIBEL", "AIRFLOW", "FREQUENCY")])

# Normalizando os dados
dados_norm <- as.data.frame(dados %>%
  mutate_if(is.numeric, scale))

View(dados_norm)
str(dados_norm)

# Funcao do Caret para divisao dos dados
split <- createDataPartition(y = dados_norm$STATUS, p = 0.7, list = FALSE)

# Criando dados de treino e de teste
dados_treino <- dados_norm[split, c("SIZE", "DISTANCE", "DESIBEL", "AIRFLOW", "FREQUENCY")]
dados_teste <- dados_norm[-split, c("SIZE", "DISTANCE", "DESIBEL", "AIRFLOW", "FREQUENCY")]
str(dados_treino)
# Criando os labels para os dados de treino e de teste
dados_treino_labels <- dados_norm[split, 7]
dados_teste_labels <- dados_norm[-split, 7]

##### Construindo um Modelo com KNN ##### 
modelo_knn_v1 <- knn(train = dados_treino, 
                     test = dados_teste,
                     cl = dados_treino_labels, 
                     k = 21)

## Avaliando e Interpretando o Modelo
# Criando uma tabela cruzada dos dados previstos x dados atuais
CrossTable(x = dados_teste_labels, y = modelo_knn_v1, prop.chisq = FALSE)


#####  Construindo um Modelo com Algoritmo Support Vector Machine (SVM) #####

# Criando dados de treino e de teste
dados_treino_svm <- dados_norm[split, ]
dados_teste_svm <- dados_norm[-split, ]

modelo_svm_v1 <- svm(STATUS ~ ., 
                     data = dados_treino_svm, 
                     type = 'C-classification', 
                     kernel = 'radial') 

# Previsões

# Previsões nos dados de treino
pred_train <- predict(modelo_svm_v1, dados_treino_svm)

# Percentual de previsões corretas com dataset de treino
mean(pred_train == dados_treino_svm$STATUS)

# Previsões nos dados de teste
pred_test <- predict(modelo_svm_v1, dados_teste_svm)

# Percentual de previsões corretas com dataset de teste
mean(pred_test == dados_teste_svm$STATUS)

# Confusion Matrix
table(pred_test, dados_teste_svm$STATUS)


##### Construindo um Modelo com Algoritmo Random Forest ##### 
# Criando o modelo
modelo_rf_v1 = rpart(STATUS ~ ., data = dados_treino_svm, control = rpart.control(cp = .0005)) 

# Previsões nos dados de teste
tree_pred = predict(modelo_rf_v1, dados_teste_svm, type='class')

# Percentual de previsões corretas com dataset de teste
mean(tree_pred==dados_teste_svm$STATUS) 

# Confusion Matrix
table(tree_pred, dados_teste_svm$STATUS)


##### Conclusão #####
# Os algoritmos de classificação apresentaram resultados próximos, com uma pequena vantagm para 
# o Algoritmo Random Forest que seria o meu escolhido.
