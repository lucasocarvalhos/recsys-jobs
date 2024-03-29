import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Tokenizer
def tokenizer(content):
  tokens = word_tokenize(content)
  return tokens

# Remover as letras únicas
def remover_letra_unica(content):
  tokens = tokenizer(content)
  sem_letra_unica = " ".join([w for w in tokens if len(w) > 1])
  return sem_letra_unica

# Remover links (deve ser feito antes da quebra de linha)
def remover_links(content):
  sem_links = re.sub('(http\S*?\\n)', '', content)
  return sem_links

# Remover quebra de linha
def remover_quebra_de_linha(content):
  sem_quebras = ' '.join(re.sub('\n', ' ', content).lower().split()[1:])
  return sem_quebras

# Remover caracteres especiais (deve ser feito antes de remover email)
def remover_caracteres(content):
  tokens = tokenizer(content)
  sem_caracteres = ' '.join([w for w in tokens if w.isalnum()])
  return sem_caracteres

# Remover emails
def remover_email(content):
  sem_email = re.sub(r'([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})', '', content)
  return sem_email

# Remover hifen
def remover_hifen(content):
  sem_hifen = re.sub(r'(?<![a-záéíóúâêôãõç])-|-(?![a-záéíóúâêôãõç])', '', content, flags=re.IGNORECASE) # substituir hífens, desde que não estejam entre palavras
  return sem_hifen

# Remover números
def remover_numeros(content):
  sem_numeros = re.sub(r'[0-9][a-z]*', '', content)
  return sem_numeros

# Remover barras
def remover_barras(content):
  sem_barras = re.sub(r'/', '', content)
  return sem_barras

# Remover espaços duplos
def remover_duplos(content):
  sem_duplos = ' '.join(content.split())
  return sem_duplos

# Remover nome de mês
def remover_meses(content):
  meses = [
      'mês',
      'meses',
      'anos',
      'janeiro',
      'fevereiro',
      'março',
      'abril',
      'maio',
      'junho',
      'julho',
      'agosto',
      'setembro',
      'outubro',
      'novembro',
      'dezembro'
  ]
  tokens = tokenizer(content)
  sem_meses = ' '.join([w for w in tokens if not w in meses])
  return sem_meses

# Remover palavras desnecessárias (presente em currículos do linkedin)
def remover_termos(content):
  termos = [
      'publications',
      'mailto',
      'page',
      'of',
      'present'
  ]
  tokens = tokenizer(content)
  sem_termos = ' '.join([w for w in tokens if not w in termos])
  return sem_termos

# Remover stopwords
def remover_stopwords(content):
  stop_words = set(stopwords.words('portuguese'))
  tokens = tokenizer(content)
  sem_stopwords = ' '.join([w for w in tokens if not w in stop_words])
  return sem_stopwords

# Realizar todos os pré-processamentos
def preprocess_txt(content):
  sem_links = remover_links(content)
  sem_quebras = remover_quebra_de_linha(sem_links)
  sem_caracteres = remover_caracteres(sem_quebras)
  sem_email = remover_email(sem_caracteres)
  sem_hifen = remover_hifen(sem_email)
  sem_numeros = remover_numeros(sem_hifen)
  sem_barras = remover_barras(sem_numeros)
  sem_duplos = remover_duplos(sem_barras)
  sem_meses = remover_meses(sem_duplos)
  sem_letras_unicas = remover_letra_unica(sem_meses)
  sem_termos = remover_termos(sem_letras_unicas)
  sem_stopwords = remover_stopwords(sem_termos)
  return sem_stopwords