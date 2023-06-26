from nltk import FreqDist
import numpy as np
from sklearn import svm

# Função para calcular o tamanho médio das palavras
def calcular_tamanho_medio_palavras(frase):
    palavras = frase.split()
    tamanho_palavras = [len(palavra) for palavra in palavras]
    tamanho_medio = np.mean(tamanho_palavras)
    return tamanho_medio

# Função para calcular a frequência das letras
def calcular_frequencia_letras(frase):
    frequencia_letras = FreqDist([letra for letra in frase if letra.isalpha()])
    return dict(frequencia_letras)

# Função para extrair as características de uma frase
def extrair_caracteristicas(frase):
    frequencias = calcular_frequencia_letras(frase)
    tamanho_medio_palavras = calcular_tamanho_medio_palavras(frase)

    caracteres_arabicos = [chr(i) for i in range(97, 123)]  # Letras minúsculas de 'a' a 'z'
    vetor_caracteristicas = []
    
    for caract in caracteres_arabicos:
        if caract in frequencias:
            vetor_caracteristicas.append(frequencias[caract])
        else:
            vetor_caracteristicas.append(0)
    
    vetor_caracteristicas.append(tamanho_medio_palavras)
    
    return vetor_caracteristicas

# Função auxiliar para criar vetor de características
def criar_vetor_caracteristicas(dados):
    max_num_caracteristicas = 26 + 1  # 26 letras do alfabeto + 1 para o tamanho médio das palavras
    matriz_caracteristicas = np.zeros((len(dados), max_num_caracteristicas))
    
    for i, dado in enumerate(dados):
        matriz_caracteristicas[i, :] = dado
    
    return matriz_caracteristicas

# Ler os arquivos de texto e criar dicionários de frases para cada idioma
dados_treinamento = {
    'portugues': [],
    'ingles': [],
    'italiano': []
}

with open('portugues.txt', 'r') as arquivo_portugues:
    for linha in arquivo_portugues:
        linha = linha.strip()
        frase = linha.replace("Frase:", "").strip()
        dados_treinamento['portugues'].append(frase)

with open('ingles.txt', 'r') as arquivo_ingles:
    for linha in arquivo_ingles:
        linha = linha.strip()
        frase = linha.replace("Frase:", "").strip()
        dados_treinamento['ingles'].append(frase)

with open('italiano.txt', 'r') as arquivo_italiano:
    for linha in arquivo_italiano:
        linha = linha.strip()
        frase = linha.replace("Frase:", "").strip()
        dados_treinamento['italiano'].append(frase)

# Treinar o modelo SVM
treinamento_dados = []
treinamento_classes = []

for classe, frases in dados_treinamento.items():
    for frase in frases:
        treinamento_dados.append(extrair_caracteristicas(frase))
        treinamento_classes.append(classe)

matriz_caracteristicas = criar_vetor_caracteristicas(treinamento_dados)

# Treinar o modelo SVM
modelo = svm.SVC()
modelo.fit(matriz_caracteristicas, treinamento_classes)

# Ler o arquivo de testes
dados_teste = []

with open('teste.txt', 'r') as arquivo_teste:
    for linha in arquivo_teste:
        linha = linha.strip()
        frase = linha.replace("Frase:", "").strip()
        dados_teste.append(frase)

# Classificar as frases de teste
for dado in dados_teste:
    caracteristicas = extrair_caracteristicas(dado)
    vetor_caracteristicas_teste = criar_vetor_caracteristicas([caracteristicas])
    resultado = modelo.predict(vetor_caracteristicas_teste)

    print("Texto:", dado)
    print("Idioma:", resultado[0])
    print("------------------------")


# Pedir ao usuário para inserir um novo texto
texto_desconhecido = input("Digite um novo texto: ")

# Extrair as características do texto desconhecido
caracteristicas_texto_desconhecido = extrair_caracteristicas(texto_desconhecido)
vetor_caracteristicas_desconhecido = criar_vetor_caracteristicas([caracteristicas_texto_desconhecido])

# Classificar o texto desconhecido
resultado_desconhecido = modelo.predict(vetor_caracteristicas_desconhecido)

# Apresentar a língua indicada pelo modelo
print("Língua indicada pelo modelo:", resultado_desconhecido[0])
