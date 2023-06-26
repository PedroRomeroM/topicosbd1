import nltk
import unicodedata
from nltk import FreqDist
import numpy as np
from sklearn import svm

# Função para remover acentos
def remover_acentos(frase):
    frase_sem_acentos = ''.join(
        (c for c in unicodedata.normalize('NFD', frase) if unicodedata.category(c) != 'Mn')
    )
    return frase_sem_acentos

# Função para calcular a frequência das letras
def calcular_frequencia_letras(frase):
    frequencia_letras = FreqDist([letra for letra in frase if letra.isalpha()])
    return dict(frequencia_letras)

# Função para extrair as características de uma frase
def extrair_caracteristicas(frase):
    frase_sem_acentos = remover_acentos(frase.lower())
    frequencias = calcular_frequencia_letras(frase_sem_acentos)
    tamanho_medio_palavras = calcular_tamanho_medio_palavras(frase_sem_acentos)
    # Adicione outras características aqui, se necessário
    return [frequencias, tamanho_medio_palavras]

# Função auxiliar para criar vetor de características
def criar_vetor_caracteristicas(dados, max_num_caracteristicas):
    matriz_caracteristicas = np.zeros((len(dados), max_num_caracteristicas))
    for i, dado in enumerate(dados):
        vetor = []
        for caract in dado:
            if isinstance(caract, dict):
                for _, freq in sorted(caract.items()):
                    vetor.append(freq)
            else:
                vetor.append(caract)
        matriz_caracteristicas[i, :len(vetor)] = vetor[:max_num_caracteristicas]

    return matriz_caracteristicas

# Função para calcular o tamanho médio das palavras
def calcular_tamanho_medio_palavras(frase):
    palavras = frase.split()
    tamanho_palavras = [len(palavra) for palavra in palavras]
    tamanho_medio = np.mean(tamanho_palavras)
    return tamanho_medio

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

max_num_caracteristicas = max(len(caract) if isinstance(caract, dict) else 1 for dado in treinamento_dados for caract in dado)

matriz_caracteristicas = criar_vetor_caracteristicas(treinamento_dados, max_num_caracteristicas)

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
    vetor_caracteristicas_teste = criar_vetor_caracteristicas([caracteristicas], max_num_caracteristicas)
    resultado = modelo.predict(vetor_caracteristicas_teste)

    print("Texto:", dado)
    print("Idioma:", resultado[0])
    print("------------------------")


# Pedir ao usuário para inserir um novo texto
texto_desconhecido = input("Digite um novo texto: ")

# Extrair as características do texto desconhecido
caracteristicas_texto_desconhecido = extrair_caracteristicas(texto_desconhecido)
vetor_caracteristicas_desconhecido = criar_vetor_caracteristicas([caracteristicas_texto_desconhecido], max_num_caracteristicas)

# Classificar o texto desconhecido
resultado_desconhecido = modelo.predict(vetor_caracteristicas_desconhecido)

# Apresentar a língua indicada pelo modelo
print("Língua indicada pelo modelo: ", resultado_desconhecido[0])

